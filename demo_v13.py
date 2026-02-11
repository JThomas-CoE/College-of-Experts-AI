"""
V13 MODULAR END-TO-END DEMO

Refactored from demo_v12_full.py with proper module separation:
- GUI components moved to src.gui
- Core classes moved to src.expert_scope, src.savant_pool, src.schedulers
- Assist resolver in src.assist_resolver

Run: python demo_v13.py [--demo | --headless]
"""

import asyncio
import json
import sys
import logging
import re
import gc
import argparse
import threading
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import gc
import traceback

# Setup logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("demo_debug.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("demo_v13")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modularized components
from src.gui import CoEDemoGUI
from src.embedding_manager import EmbeddingManager
from src.template_validator import TemplateValidator
from src.template_adapter import TemplateAdapter
from src.deepseek_parser import strip_thinking  # Used by robust_strip_thinking
from src.knowledge_layer import (
    KnowledgeRetriever,
    initialize_knowledge_base
)
from src.quality_gate import QualityGate
# REMOVED: persona_context and prompt_templates - replaced by PromptCompiler
from src.vram_manager import VRAMAwareScheduler, create_scheduler
from src.framework_scheduler import TaskFramework, FrameworkSlot
from src.working_memory import WorkingMemory
from src.expert_scope import ExpertScope, EXPERT_SCOPES, get_expert_scope  # ExpertScope used in type hints
from src.savant_pool import SavantPool
from src.assist_resolver import AssistResolver
from src.schedulers import DAGScheduler
from src.memory_router import MemoryVectorRouter, create_memory_router  # create_memory_router used in run_demo
from src.query_classifier import QueryClassifier, QueryTier, ClassificationResult
from src.prompt_compiler import PromptCompiler, CompiledPlan

# Configuration
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 48000,
    "max_expert_slots": 3,
    "deepseek_slot_cost": 2,
    "embedding_mb": 2000,
    "base_model_mb": 5000,
    "deepseek_base_mb": 5000,
}

# Savant model mappings
SAVANT_MODELS = {
    "python_backend": "models/Qwen2.5-Coder-7B-DML",
    "sql_schema_architect": "models/Qwen2.5-Coder-7B-DML",
    "html_css_specialist": "models/Qwen2.5-Coder-7B-DML",
    "math_expert": "models/Qwen2.5-Math-7B-DML",
    "security_architect": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
    "legal_contracts": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
}

LAMBDA_EXCLUSION_PENALTY = 0.3

# Query processing configuration - all tunable thresholds in one place
QUERY_CONFIG = {
    # Temperature
    "min_temperature": 0.01,  # Minimum clamp to avoid deterministic issues
    
    # Template matching thresholds
    "template_exact_match": 0.95,     # Use template directly without adaptation
    "template_adapt_threshold": 0.75,  # Adapt template vs decompose from scratch
    
    # Output validation
    "min_output_length": 50,  # Minimum chars for valid DeepSeek output
    
    # Memory router
    "memory_similarity_threshold": 0.85,  # Cache hit threshold
    
    # Knowledge retrieval  
    "knowledge_max_tokens": 1200,  # Max tokens for knowledge context
}


# NOTE: QueryTier enum and classification logic moved to src/query_classifier.py
# The new QueryClassifier uses LLM-based assessment for robust complexity detection.


# Output control flags
SHOW_DECOMPOSITION = True
SHOW_ROUTING = True
SHOW_QUALITY_GATE = True
SHOW_PROMPTS = True
SHOW_THINKING_TOKENS = False
SHOW_OUTPUT_PREVIEW = True  # Show first 800 chars of each slot's output for quality verification


class DualEmbeddingRouter:
    """Routes tasks using dual embeddings: net_score = capability_score - λ * exclusion_score"""
    
    def __init__(self, embedding_manager: EmbeddingManager, expert_scopes: Dict[str, ExpertScope]):
        self.embedding_manager = embedding_manager
        self.expert_scopes = expert_scopes
        self.expert_ids = list(expert_scopes.keys())
        
        # Pre-compute embeddings
        print("[DualRouter] Computing expert capability embeddings...")
        self.capability_vecs = np.array([
            embedding_manager.encode(scope.capability_scope)
            for scope in expert_scopes.values()
        ])
        
        print("[DualRouter] Computing expert exclusion embeddings...")
        self.exclusion_vecs = np.array([
            embedding_manager.encode(scope.exclusion_scope)
            for scope in expert_scopes.values()
        ])
        
        # Normalize
        self.capability_vecs = self.capability_vecs / (np.linalg.norm(self.capability_vecs, axis=1, keepdims=True) + 1e-8)
        self.exclusion_vecs = self.exclusion_vecs / (np.linalg.norm(self.exclusion_vecs, axis=1, keepdims=True) + 1e-8)
        
        print(f"[DualRouter] Indexed {len(self.expert_ids)} experts")
    
    def route(self, task_text: str, lambda_penalty: float = LAMBDA_EXCLUSION_PENALTY) -> Tuple[str, float, Dict]:
        """Route task to best expert using dual-embedding scoring."""
        task_vec = self.embedding_manager.encode(task_text)
        task_vec = task_vec / (np.linalg.norm(task_vec) + 1e-8)
        
        cap_scores = self.capability_vecs @ task_vec
        exc_scores = self.exclusion_vecs @ task_vec
        net_scores = cap_scores - lambda_penalty * exc_scores
        
        best_idx = np.argmax(net_scores)
        best_expert = self.expert_ids[best_idx]
        
        debug = {
            "net_scores": {eid: float(net_scores[i]) for i, eid in enumerate(self.expert_ids)},
            "top_3": sorted(
                [(eid, float(net_scores[i])) for i, eid in enumerate(self.expert_ids)],
                key=lambda x: -x[1]
            )[:3]
        }
        
        return best_expert, float(net_scores[best_idx]), debug


class ModelExecutor:
    """Executor using OGA models with VRAM-aware scheduling."""
    
    def __init__(self, scheduler: VRAMAwareScheduler, savant_pool: SavantPool):
        self.scheduler = scheduler
        self.savant_pool = savant_pool
        self.current_savant: Optional[str] = None
        self._slot_id_counter = 0
        self._slot_savant_map: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def _get_next_slot_id(self) -> str:
        self._slot_id_counter += 1
        return f"exec_slot_{self._slot_id_counter}"
    
    def ensure_loaded(self, savant_id: str, model_path: str, slot_id: str = None) -> str:
        """
        Ensure model is loaded using VRAM-aware scheduler.
        
        Args:
            savant_id: Identifier for the savant model
            model_path: Path to the model files
            slot_id: Optional slot identifier. If None, auto-generates unique ID.
                    If provided, must not already exist in slot map.
        
        Returns:
            slot_id: The slot identifier for this reservation
            
        Raises:
            ValueError: If provided slot_id already exists with different savant
            RuntimeError: If VRAM allocation fails
        """
        with self._lock:
            if slot_id is None:
                slot_id = self._get_next_slot_id()
            elif slot_id in self._slot_savant_map:
                # Slot already exists - check if it's the same savant
                existing_savant = self._slot_savant_map[slot_id]
                if existing_savant == savant_id:
                    logger.warning(f"Slot {slot_id} already mapped to {savant_id}, reusing")
                    return slot_id
                else:
                    raise ValueError(
                        f"Slot {slot_id} already mapped to {existing_savant}, "
                        f"cannot remap to {savant_id}. Use a different slot_id or None for auto-generation."
                    )
            
            self._slot_savant_map[slot_id] = savant_id
            
            is_deepseek = "deepseek" in savant_id.lower()
            model_size_mb = CUSTOM_VRAM_CONFIG["deepseek_base_mb"] if is_deepseek else CUSTOM_VRAM_CONFIG["base_model_mb"]
            
            # Load if not already loaded
            if not self.savant_pool.is_loaded(savant_id):
                self.savant_pool.load(savant_id, model_path, model_size_mb)
            
            # Reserve VRAM via scheduler
            model = self.scheduler.start_slot(slot_id, savant_id, model_size_mb)
            
            if model is None:
                # Cleanup on failure
                del self._slot_savant_map[slot_id]
                raise RuntimeError(f"Cannot allocate VRAM for {savant_id}")
            
            self.current_savant = savant_id
            return slot_id
    
    def complete_slot(self, slot_id: str):
        """Complete slot and release VRAM."""
        if slot_id and slot_id in self._slot_savant_map:
            savant_id = self._slot_savant_map[slot_id]
            self.scheduler.complete_slot(slot_id, savant_id)
            
            # Unload if refcount is 0
            if savant_id in self.savant_pool.loaded_info:
                info = self.savant_pool.loaded_info[savant_id]
                if info.refcount == 0:
                    self.savant_pool.unload(savant_id)
            
            del self._slot_savant_map[slot_id]
    
    async def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048, slot_id: str = None) -> str:
        """Generate response with OGA.
        
        If slot_id is provided, uses the savant associated with that slot.
        Otherwise falls back to current_savant.
        """
        import onnxruntime_genai as og
        
        # Determine which savant to use based on slot_id (CRITICAL for VRAM efficiency)
        if slot_id and slot_id in self._slot_savant_map:
            savant_id = self._slot_savant_map[slot_id]
        elif self.current_savant:
            savant_id = self.current_savant
        else:
            raise RuntimeError("No savant loaded and no slot_id provided")
        
        savant_info = self.savant_pool.loaded_info.get(savant_id)
        if savant_info is None:
            raise RuntimeError(f"Savant {savant_id} not found in loaded pool")
        
        # Log model reuse for debugging
        logger.debug(f"[ModelExecutor] Using savant: {savant_id} (slot_id={slot_id})")
        
        # Get model and tokenizer from the correct savant
        model = savant_info.model
        tokenizer = self.savant_pool.get_tokenizer(savant_id)
        
        # Wrap in ChatML format for Qwen models
        # Without this, the model produces garbage output
        is_deepseek = "deepseek" in savant_id.lower()
        if is_deepseek:
            # DeepSeek uses same ChatML format
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Qwen2.5-Coder uses ChatML format
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_tokens = tokenizer.encode(formatted_prompt)
        input_len = len(input_tokens)
        
        # [Context Safety] Dynamically clamp max_tokens to model limit
        context_limit = self.savant_pool.get_context_length(savant_id)
        # Reserve buffer (50 tokens)
        max_capacity = context_limit - input_len - 50
        
        if max_capacity < 128:
            logger.warning(f"Context saturated! Input: {input_len}, Limit: {context_limit}. Truncating generation.")
            effective_max_tokens = 128 # Force small output
        else:
            effective_max_tokens = min(max_tokens, max_capacity)
            
        if effective_max_tokens < max_tokens:
            logger.info(f"Clamped generation tokens: {max_tokens} -> {effective_max_tokens} (Context: {context_limit})")

        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=input_len + effective_max_tokens,
            temperature=max(temperature, QUERY_CONFIG["min_temperature"])
        )
        
        generator = None
        try:
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)
            
            output_tokens = []
            for _ in range(effective_max_tokens):
                if generator.is_done():
                    break
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                output_tokens.append(token)
            
            response = tokenizer.decode(output_tokens)
            return response.strip()
            
        finally:
            if generator:
                del generator
            del params
    
    def get_vram_status(self) -> Dict:
        return self.scheduler.get_status()


def build_slot_prompt(slot: FrameworkSlot, query: str, expert_scope: ExpertScope, 
                      expert_id: str, dependency_signatures: str = "", 
                      context_budget: int = 4096, knowledge_context: str = "") -> str:
    """Build persona-aware prompt for a slot."""
    compact_mode = context_budget <= 4096
    
    # Extract project context
    project_context = query.strip().split('\n')[0].strip()
    
    # Load persona
    persona_context = get_persona_for_expert(expert_id)
    output_formats = ", ".join(slot.expected_outputs) if slot.expected_outputs else None
    
    # Build prompt
    prompt = build_persona_aware_prompt(
        persona=persona_context,
        task_description=slot.description,
        project_context=project_context,
        output_formats=output_formats,
        dependency_signatures=dependency_signatures if slot.dependencies else None,
        knowledge_context=knowledge_context,
        compact_mode=compact_mode,
        context_budget=context_budget
    )
    
    return prompt


def validate_code_blocks(content: str) -> Tuple[bool, List[str], str]:
    """Validate that all code blocks are properly closed. Auto-close if needed."""
    issues = []
    
    code_block_starts = len(re.findall(r'```\w+', content))
    code_block_ends = content.count('```') - code_block_starts
    
    if code_block_starts > code_block_ends:
        missing = code_block_starts - code_block_ends
        issues.append(f"Auto-closed {missing} unclosed code block(s)")
        # Auto-close missing blocks
        content = content.rstrip() + '\n' + '\n```\n' * missing
        return True, issues, content
    
    return True, issues, content


def robust_strip_thinking(text: str) -> Tuple[str, Optional[str]]:
    """Enhanced thinking stripping for DeepSeek responses."""
    cleaned, thinking = strip_thinking(text)
    
    if '</think>' in cleaned:
        parts = cleaned.split('</think>')
        thinking = parts[0] if thinking is None else thinking + "\n" + parts[0]
        cleaned = '</think>'.join(parts[1:]).strip()
    
    if cleaned.startswith(('Alright,', 'Okay,', 'Let me', "I'll", "First,")):
        markers = ['\n## ', '\n# ', '\n```', '\n🔐', '\n---', '\nI\'m sorry']
        earliest_pos = len(cleaned)
        for marker in markers:
            pos = cleaned.find(marker)
            if 0 < pos < earliest_pos:
                earliest_pos = pos
        
        if earliest_pos < len(cleaned):
            thinking_part = cleaned[:earliest_pos].strip()
            cleaned = cleaned[earliest_pos:].strip()
            thinking = thinking_part if thinking is None else thinking + "\n" + thinking_part
    
    return cleaned, thinking


async def run_demo(gui: Optional[CoEDemoGUI] = None, query_override: Optional[str] = None, components: Dict[str, Any] = None) -> bool:
    """Run V13 demo.
    
    Args:
        gui: Optional GUI instance for progress display
        query_override: Optional custom query (uses default if None)
        
    Returns:
        True if demo completed successfully, False otherwise
    """
    
    def gui_progress(msg: str) -> None:
        if gui:
            gui.show_progress(msg)
    
    print("\n" + "="*70)
    print(" College of Experts V13 - Modular Architecture")
    print("="*70)
    gui_progress("Initializing...")
    
    # Initialize components
    print("\n[1] Initializing components...")
    
    if components:
        print("\n[1] Using shared components (VRAM optimized)...")
        embedding_manager = components["embedding_manager"]
        router = components["router"]
        scheduler = components["scheduler"]
        savant_pool = components["savant_pool"]
        executor = components["executor"]
    else:
        import torch
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Device: {embed_device}")
        
        embedding_manager = EmbeddingManager(model_name="BAAI/bge-m3", device=embed_device)
        print("    ✓ Embedding manager")
        
        router = DualEmbeddingRouter(embedding_manager, EXPERT_SCOPES)
        print("    ✓ Router")
        
        scheduler = create_scheduler(profile_name="balanced")
        savant_pool = SavantPool(scheduler_pool=scheduler.savant_pool)
        executor = ModelExecutor(scheduler, savant_pool)
        print("    ✓ VRAM scheduler & executor")
        
    # [VRAM GUARD] Force safe concurrency limit
    if hasattr(scheduler, "max_parallel"):
         print(f"    [VRAM Guard] Enforcing max_parallel=2 (was {scheduler.max_parallel})")
         scheduler.max_parallel = 2
    
    # Set up DeepSeek auto-reload callback for fast follow-up queries
    def _on_all_models_unloaded():
        """Callback: reload DeepSeek when all savant models are unloaded."""
        print("    [DeepSeek Auto-Reload] All savant models unloaded - reloading DeepSeek...")
        
        # Check VRAM availability
        vram_status = scheduler.get_status()
        vram_budget_mb = vram_status.get("vram", {}).get("budget_mb", 48000)
        current_usage_mb = vram_status.get("vram", {}).get("current_usage_mb", 0)
        available_mb = vram_budget_mb - current_usage_mb
        
        # DeepSeek needs ~5GB
        deepseek_size_mb = CUSTOM_VRAM_CONFIG["deepseek_base_mb"]
        
        if available_mb >= deepseek_size_mb:
            try:
                # Use the same model path as security_architect (both use DeepSeek)
                deepseek_model_path = SAVANT_MODELS.get("security_architect")
                if deepseek_model_path:
                    savant_pool.load("deepseek_r1", deepseek_model_path, deepseek_size_mb)
                    print(f"    [DeepSeek Auto-Reload] DeepSeek ready for follow-up queries")
            except Exception as e:
                print(f"    [DeepSeek Auto-Reload] Failed to reload: {e}")
        else:
            print(f"    [DeepSeek Auto-Reload] Insufficient VRAM ({available_mb}MB available, need {deepseek_size_mb}MB)")
    
    # savant_pool.set_all_unloaded_callback(_on_all_models_unloaded)
    # print("    ✓ DeepSeek auto-reload callback configured")
    print("    [Info] DeepSeek auto-reload callback DISABLED for stability")
    
    if components and "memory" in components:
        quality_gate = components["quality_gate"]
        assist_resolver = components["assist_resolver"]
        memory = components["memory"]
        validator = components["validator"]
        print("    ✓ Memory/Helpers reused from shared components")
    else:
        quality_gate = QualityGate()
        assist_resolver = AssistResolver()
        memory = WorkingMemory()
        validator = TemplateValidator()
        print("    ✓ Quality gate, resolver, memory, validator created")
    
    if components and "memory_router" in components:
        memory_router = components["memory_router"]
        print("    ✓ Memory Vector Router reused (Persistent Index)")
    else:
        # Memory-first router - checks WorkingMemory before routing to experts
        memory_router = create_memory_router(
            working_memory=memory,
            vector_router=router,  # Uses DualEmbeddingRouter for fallback
            embedding_manager=embedding_manager,
            config={"similarity_threshold": QUERY_CONFIG["memory_similarity_threshold"]}
        )
        print("    ✓ Memory Vector Router created")
    
    # Knowledge base
    knowledge_retriever = KnowledgeRetriever(
        embedding_fn=embedding_manager.encode,
        knowledge_base_dir="data/knowledge",
        enable_web_search=False
    )
    
    if len(knowledge_retriever.local_kb.chunks) == 0:
        initialize_knowledge_base(embedding_manager.encode, "data/knowledge")
        knowledge_retriever.local_kb._load()
    
    knowledge_retriever.set_working_memory(memory)
    print("    ✓ Knowledge base")
    
    # Load DeepSeek router at startup - ready for Q&A immediately
    print("    Loading DeepSeek router...")
    deepseek_slot = executor.ensure_loaded("deepseek_r1", SAVANT_MODELS.get("security_architect"))
    print("    ✓ DeepSeek router loaded (ready for Q&A)")
    
    # Get query
    query = query_override or """Build a personal task manager web app with:
1. A standalone HTML file with embedded CSS and JavaScript for the frontend UI (dark theme, modern look)
2. SQLite database schema for tasks (id, title, description, priority, due_date, completed, created_at)
3. Python Flask backend with REST API endpoints for CRUD operations on tasks
4. Password hashing for user authentication using bcrypt"""
    
    print(f"\n[2] Query:\n{query[:200]}...")
    gui_progress("Analyzing query...")
    
    # === TIER 1 CHECKS FIRST ===
    
    # 1. Memory Check
    print("    Checking memory for existing answers...")
    route_result = memory_router.route(query)
    
    if route_result["source"] == "memory":
        print(f"    ✓ Memory hit (Confidence: {route_result['confidence']:.2f})")
        content = route_result["content"]
        expert_id = route_result["expert_id"]
        
        gui_progress("Found answer in memory...")
        if gui:
            gui.show_section(f"Memory Result ({expert_id})", content, expert_id)
            gui.generation_complete()
            gui.set_status(f"Done (from memory)!")
            
        print("\n--- Memory Result ---")
        print(content[:500] + "..." if len(content) > 500 else content)
        return True

    # 2. Classifier Check
    print("    Classifying query complexity...")
    classifier = QueryClassifier(executor, deepseek_slot, router=memory_router)
    classification = await classifier.classify(query)
    
    print(f"    Classification: {classification.tier.name} ({classification.reasoning})")
    
    if classification.tier == QueryTier.TIER1_TRIVIAL:
        # Trivial queries answered directly by DeepSeek (no framework)
        print(f"    Trivial query -> DeepSeek direct answer")
        gui_progress("Answering directly...")
        
        # Inject memory context if available
        context_str = ""
        completed = memory.get_all_completed()
        if completed:
            context_str = "CONTEXT OF CURRENT PROJECT (Generated Code):\n"
            for sid in completed:
                sres = memory.get(sid)
                if sres:
                    context_str += f"- {sid} (by {sres.expert_id})\n"
            context_str += "\nThe user is asking about the code above.\n"

        # Use suppress_thinking flag from metadata
        if context_str:
            base_prompt = f"{context_str}\nUser Question: {query}"
        else:
            base_prompt = f"Answer this question concisely: {query}"
            
        prompt = base_prompt
        if classification.metadata.get('suppress_thinking', False):
            prompt = f"<|im_start|>system\nYou are a helpful assistant. Answer directly without showing your thinking process.<|im_end|>\n<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = await executor.generate(
            prompt,
            temperature=0.3,
            max_tokens=4096,
            slot_id=deepseek_slot
        )
        cleaned, _ = robust_strip_thinking(response)
        
        if gui:
            gui.show_section("Answer", cleaned, "deepseek_r1")
        # NOTE: Do NOT unload DeepSeek here, keep resident for supervisor role
        # executor.complete_slot(deepseek_slot)
        
        if gui:
            gui.generation_complete()
            gui.set_status("Done!")
        return True

    # === TIER 2/3: TEMPLATE MATCHING (Only for Complex Queries) ===
    
    # [VRAM SHIELD] Aggressive cleanup before heavy lifting
    print("    [VRAM Shield] Unloading inactive experts before complex task...")
    try:
        # 1. Unload all except DeepSeek (Supervisor)
        if savant_pool:
             for sid in list(savant_pool.loaded_info.keys()):
                 if "deepseek" not in sid.lower() and "reasoner" not in sid.lower():
                     savant_pool.unload(sid)
        
        # 2. Hard GC
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"    [Warning] VRAM Shield error: {e}")

    # Load templates and pre-compute embeddings
    print(f"\n[Tier 2/3] Loading templates for complex task...")
    
    template_dir = Path("config/framework_templates")
    with open(template_dir / "all_templates.json") as f:
        templates = json.load(f)
    
    # Pre-compute template embeddings
    print(f"    Computing embeddings for {len(templates)} templates...")
    template_embeddings = {}
    for template in templates:
        template_id = template.get("id", "unknown")
        text = f"{template.get('title', '')}. {template.get('description', '')}"
        vec = embedding_manager.encode(text)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        template_embeddings[template_id] = vec
    
    # Rank templates
    query_vec = embedding_manager.encode(query)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    
    ranked = []
    for template in templates:
        template_id = template.get("id", "unknown")
        vec = template_embeddings.get(template_id)
        if vec is None:
            text = f"{template.get('title', '')}. {template.get('description', '')}"
            vec = embedding_manager.encode(text)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        sim = float(np.dot(query_vec, vec))
        ranked.append((sim, template))
    
    ranked.sort(key=lambda x: x[0], reverse=True)
    print(f"    Best match: {ranked[0][1]['id']} ({ranked[0][0]:.3f})")
    
    # Adapt template
    gui_progress("Using DeepSeek for template adaptation...")
    
    adapter = TemplateAdapter(
        executor=executor,
        validator=validator,
        prompt_template_path="config/prompts/template_adapter.yaml"
    )
    
    best_similarity, best_template = ranked[0]
        
    if classification.tier == QueryTier.TIER2_SINGLE:
        # Single expert problem -> one slot framework
        expert_id = classification.primary_expert or "python_backend"
        title = f"{expert_id.replace('_', ' ').title()} Task"
        print(f"    Single expert problem -> {expert_id}")
        framework = TaskFramework(
            id=f"single_{int(datetime.now().timestamp())}",
            title=title,
            description=query,
            slots=[
                FrameworkSlot(
                    id=expert_id,
                    title=title,
                    description=query,
                    persona=expert_id,
                    dependencies=[],
                    expected_outputs=["Solution"]
                )
            ]
        )
    elif best_similarity > QUERY_CONFIG["template_exact_match"]:
        print("    Using template directly (high similarity)")
        framework = TaskFramework.from_dict(best_template)
    elif best_similarity > QUERY_CONFIG["template_adapt_threshold"]:
        print("    Adapting template...")
        result = await adapter.adapt(query, best_template, list(EXPERT_SCOPES.keys()), best_similarity)
        framework = result.framework if result and result.framework else TaskFramework.from_dict(best_template)
    else:
        print("    Decomposing from scratch...")
        result = await adapter.decompose_from_scratch(query, list(EXPERT_SCOPES.keys()))
        framework = result.framework if result and result.framework else adapter.create_single_slot_fallback(query).framework
    
    print(f"    Framework: {len(framework.slots)} slots")
    
    # =========================================================================
    # PREPARATION PHASE: Compile all prompts while DeepSeek is loaded
    # - Batch retrieve knowledge for all slots
    # - Generate optimized prompts with knowledge included
    # - Store in RAM (prompts are just text, no VRAM cost)
    # =========================================================================
    print("\n[3b] Compiling prompts...")
    gui_progress("Compiling prompts with knowledge...")
    
    # PromptCompiler builds structured prompts directly (no LLM needed)
    prompt_compiler = PromptCompiler(knowledge_retriever=knowledge_retriever)
    
    # Compile all slots - this generates optimized prompts with knowledge
    compiled_plan = await prompt_compiler.compile(
        query=query,
        slots=framework.slots,
        max_knowledge_tokens=QUERY_CONFIG["knowledge_max_tokens"]
    )
    
    print(f"    Compiled {len(compiled_plan.slots)} prompts")
    print(f"    Execution order: {' -> '.join(compiled_plan.execution_order)}")
    
    # NOW unload DeepSeek - prompts are safely in RAM
    executor.complete_slot(deepseek_slot)
    
    # Show decomposition
    if SHOW_DECOMPOSITION:
        print("\n    Slots:")
        for slot in framework.slots:
            deps = f" (deps: {', '.join(slot.dependencies)})" if slot.dependencies else ""
            print(f"      - {slot.id}: {slot.title}{deps}")
    
    # Execute
    gui_progress(f"Executing {len(framework.slots)} slots...")
    print(f"\n[4] Executing {len(framework.slots)} slots...")
    
    # [VRAM GUARD] Force serial execution to prevent overload
    dag = DAGScheduler(framework.slots, max_concurrent=1)
    
    async def execute_slot(slot_id: str) -> Tuple[str, bool]:
        """
        Execute a single slot with guaranteed cleanup.
        
        Two-Phase Execution Pattern:
        
        PHASE 1 - Domain Expert (2 attempts):
            1. Route task to specialized domain expert (python, sql, etc.)
            2. Generate response with expert's recommended temperature
            3. Validate with quality gate (completeness + accuracy)
            4. On failure: refine prompt and retry once
            
        PHASE 2 - DeepSeek Fallback (if Phase 1 fails):
            1. Unload domain expert to free VRAM
            2. Wait for unload confirmation (max 5s timeout)
            3. Verify VRAM availability for DeepSeek
            4. Load DeepSeek-R1 as "supervisor of last resort"
            5. Generate best-effort response with full context
            6. Accept output if length >= min_output_length (50 chars)
        
        VRAM Management:
            - exec_slot: Tracks domain expert's slot ID
            - deepseek_exec_slot: Tracks DeepSeek's slot ID (if escalated)
            - Both are guaranteed to be cleaned up in finally block
        
        Args:
            slot_id: ID of the framework slot to execute
            
        Returns:
            Tuple[str, bool]: (slot_id, success_flag)
        """
        slot = dag.slots[slot_id]
        print(f"\n  * {slot_id}: {slot.title}", end=" ")
        
        exec_slot = None
        deepseek_exec_slot = None
        
        try:
            # Route using memory-first router (checks WorkingMemory before experts)
            route_result = memory_router.route(slot.description, context=slot.title)
            
            if route_result["source"] == "memory":
                # Cache hit - use stored result
                print(f"[MEMORY HIT: {route_result['match_info']['slot_id']}] ✓", end=" ")
                memory.store(slot_id, route_result["expert_id"], route_result["content"])
                # Show cached result in GUI immediately
                if gui:
                    gui.show_section(f"{slot.title} (cached)", route_result["content"], route_result["expert_id"])
                gui_progress(f"{slot.title} ✓ (cached)")
                return (slot_id, True)
            
            # Memory miss - route to expert
            expert_id = route_result["expert_id"]
            expert_scope = get_expert_scope(expert_id)
            print(f"-> {expert_id}", end=" ")
            
            model_path = SAVANT_MODELS.get(expert_id, SAVANT_MODELS["python_backend"])
            gui_progress(f"Loading {expert_id}...")
            exec_slot = executor.ensure_loaded(expert_id, model_path, slot_id)
            
            # Get dependency outputs for placeholder filling
            dependency_outputs = {}
            if slot.dependencies:
                for dep_id in slot.dependencies:
                    dep_result = memory.get(dep_id)
                    if dep_result:
                        # Get stored raw content for dependency
                        dependency_outputs[dep_id] = dep_result.raw_content
                        logger.debug(f"[Dependency] {dep_id}: {len(dep_result.raw_content)} chars")
                    else:
                        logger.warning(f"[Dependency] {dep_id}: NOT FOUND in memory!")
            
            # Get pre-compiled prompt (already has knowledge embedded)
            # Just fill in dependency placeholders with actual outputs
            prompt = compiled_plan.get_prompt(slot_id, dependency_outputs)
            
            if SHOW_PROMPTS:
                print(f"\n{'='*80}\nCOMPILED PROMPT FOR {slot.title} ({len(prompt)} chars):\n{'='*80}")
                # Show dependency info
                if dependency_outputs:
                    print(f"[Dependencies filled: {list(dependency_outputs.keys())}]")
                print(prompt)
                print(f"{'='*80}\n")
            
            # PHASE 1: Try 2 times with domain expert + refined prompts
            current_prompt = prompt
            domain_retries = 2
            last_validation = None
            
            for attempt in range(domain_retries):
                try:
                    response = await executor.generate(current_prompt, max_tokens=8192, temperature=expert_scope.recommended_temp, slot_id=exec_slot)
                    cleaned, thinking = robust_strip_thinking(response)
                    
                    if SHOW_THINKING_TOKENS and thinking:
                        print(f"\n  [Thinking: {thinking[:200]}...]")
                    
                    # Validate code blocks (auto-close if needed)
                    code_ok, code_issues, cleaned = validate_code_blocks(cleaned)
                    if code_issues:
                        print(f"  [{code_issues[0]}]", end=" ")
                    
                    # Quality gate - uses same model via slot_id (no additional VRAM)
                    print("[QG", end="", flush=True)  # QG = Quality Gate start
                    validation = await quality_gate.validate_with_llm(executor, cleaned, slot, expert_id, slot_id=exec_slot)
                    last_validation = validation
                    
                    print(f":{validation.completeness_score}/{validation.accuracy_score}]", end=" ", flush=True)
                    
                    # Check passed status - code_ok AND quality gate must pass
                    if validation.passed and code_ok:
                        resolved, _ = assist_resolver.resolve_content(cleaned)
                        memory.store(slot_id, expert_id, resolved)
                        memory_router.index_slot(slot_id, resolved)
                        print(f"✓ PASS (domain expert, attempt {attempt + 1}/{domain_retries})")
                        
                        # Show output preview for quality verification
                        if SHOW_OUTPUT_PREVIEW:
                            preview = cleaned[:800] if len(cleaned) > 800 else cleaned
                            print(f"\n    [OUTPUT PREVIEW - {len(cleaned)} chars total]:")
                            print(f"    {'─'*70}")
                            for line in preview.split('\n')[:15]:  # First 15 lines
                                print(f"    {line[:100]}")
                            if len(cleaned) > 800 or len(cleaned.split('\n')) > 15:
                                print(f"    ... (truncated)")
                            print(f"    {'─'*70}")
                        
                        # Show intermediate result in GUI immediately
                        if gui:
                            gui.show_section(f"{slot.title}", resolved, expert_id)
                        gui_progress(f"{slot.title} ✓")
                        return (slot_id, True)
                    else:
                        # Determine specific failure reason
                        if validation.issues:
                            issue_msg = validation.issues[0]
                        else:
                            issue_msg = 'Quality check failed'
                        
                        if attempt < domain_retries - 1:
                            print(f"✗ FAIL (attempt {attempt + 1}/{domain_retries}): {issue_msg}")
                            current_prompt = quality_gate.generate_refined_prompt(
                                current_prompt, slot, validation.issues, validation.refined_prompt
                            )
                            print(f"  [Retrying with refined prompt...]")
                        else:
                            print(f"✗ FAIL (domain expert exhausted): {issue_msg}")
                            
                except Exception as e:
                    logger.error(f"Domain expert error: {e}\n{traceback.format_exc()}")
                    if attempt < domain_retries - 1:
                        print(f"✗ ERROR (attempt {attempt + 1}/{domain_retries}): {e}")
                    else:
                        print(f"✗ ERROR (domain expert exhausted): {e}")
            
            # PHASE 2: Domain expert failed - fallback to DeepSeek for best-effort
            print(f"  [→ Escalating to DeepSeek for best-effort...]")
            
            # Complete current expert slot to free VRAM (will be cleaned in finally too)
            if exec_slot:
                executor.complete_slot(exec_slot)
                exec_slot = None  # Mark as completed
            
            # Force unload domain expert to ensure DeepSeek doesn't overlap
            try:
                if executor.savant_pool.is_loaded(expert_id):
                    logger.info(f"Unloading {expert_id} to free VRAM for DeepSeek...")
                    executor.savant_pool.unload(expert_id)
                    
                    # Wait for unload confirmation (max 5 seconds)
                    max_wait = 5.0
                    waited = 0.0
                    while executor.savant_pool.is_loaded(expert_id) and waited < max_wait:
                        await asyncio.sleep(0.1)
                        waited += 0.1
                    
                    if executor.savant_pool.is_loaded(expert_id):
                        logger.warning(f"Failed to unload {expert_id} after {max_wait}s, proceeding anyway")
            except Exception as e:
                logger.error(f"Error unloading {expert_id}: {e}")
            
            # Explicit GC before fallback to prevent fragmentation
            gc.collect()
            
            # Verify VRAM availability before loading DeepSeek
            vram_status = executor.get_vram_status()
            available_mb = vram_status["vram"]["budget_mb"] - vram_status["vram"]["current_usage_mb"]
            deepseek_size = CUSTOM_VRAM_CONFIG["deepseek_base_mb"]
            
            if available_mb < deepseek_size:
                logger.error(f"Insufficient VRAM for DeepSeek: {available_mb}MB available, need {deepseek_size}MB")
                return (slot_id, False)
            
            # Load DeepSeek
            gui_progress(f"Loading DeepSeek for {slot.title}...")
            deepseek_exec_slot = executor.ensure_loaded("deepseek_r1", SAVANT_MODELS.get("security_architect"), f"{slot_id}_deepseek")
            
            # Build DeepSeek prompt with full context
            deepseek_prompt = f"""You are the supervisor of last resort. A domain expert failed to complete this task adequately.

TASK: {slot.description}

The previous attempts had these issues:
{chr(10).join(f"- {issue}" for issue in (last_validation.issues if last_validation else ["Quality check failed"]))}

Please provide your BEST EFFORT complete implementation. Focus on:
1. Addressing the issues mentioned above
2. Providing complete, working code
3. Meeting all requirements in the task description

{prompt}"""
            
            try:
                response = await executor.generate(deepseek_prompt, max_tokens=8192, temperature=0.3, slot_id=deepseek_exec_slot)
                cleaned, thinking = robust_strip_thinking(response)
                
                # Validate but accept even if not perfect (best effort)
                code_ok, _, cleaned = validate_code_blocks(cleaned)
                print("[QG", end="", flush=True)
                validation = await quality_gate.validate_with_llm(executor, cleaned, slot, "deepseek_r1", slot_id=deepseek_exec_slot)
                
                print(f":{validation.completeness_score}/{validation.accuracy_score}]", end=" ", flush=True)
                
                if len(cleaned.strip()) >= QUERY_CONFIG["min_output_length"]:  # Minimum sanity check
                    resolved, _ = assist_resolver.resolve_content(cleaned)
                    memory.store(slot_id, "deepseek_r1", resolved)
                    memory_router.index_slot(slot_id, resolved)
                    print(f"✓ BEST EFFORT (DeepSeek)")
                    # Show intermediate result in GUI immediately
                    if gui:
                        gui.show_section(f"{slot.title} (DeepSeek)", resolved, "deepseek_r1")
                    gui_progress(f"{slot.title} ✓ (DeepSeek best effort)")
                    return (slot_id, True)
                else:
                    print(f"✗ DeepSeek output too short")
                    
            except Exception as e:
                logger.error(f"DeepSeek error: {e}\n{traceback.format_exc()}")
                print(f"✗ DeepSeek error: {e}")
            
            return (slot_id, False)
            
        finally:
            # GUARANTEED CLEANUP: Always complete slots to prevent VRAM leaks
            if exec_slot:
                try:
                    executor.complete_slot(exec_slot)
                except Exception as e:
                    logger.error(f"Error completing exec_slot {exec_slot}: {e}")
            
            if deepseek_exec_slot:
                try:
                    executor.complete_slot(deepseek_exec_slot)
                except Exception as e:
                    logger.error(f"Error completing deepseek_exec_slot {deepseek_exec_slot}: {e}")
    
    # Execute in parallel waves
    try:
        while not dag.is_done():
            ready = dag.get_ready_slots()
            
            if not ready:
                if dag.in_progress:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    print("  [!] Stuck - dependency failure")
                    break
            
            # Start ready slots
            to_start = [s for s in ready if dag.can_start_more()][:3]
            for slot_id in to_start:
                dag.start_slot(slot_id)
            
            if len(to_start) > 1:
                tasks = [execute_slot(sid) for sid in to_start]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"\n  [!] Slot execution error: {result}")
                        logger.error(f"Slot execution error:\n{traceback.format_exception(type(result), result, result.__traceback__)}")
                        dag.complete_slot(to_start[i], False)
                    else:
                        slot_id, success = result
                        dag.complete_slot(slot_id, success)
            else:
                slot_id = to_start[0]
                try:
                    _, success = await execute_slot(slot_id)
                    dag.complete_slot(slot_id, success)
                except Exception as e:
                    print(f"\n  [!] Slot {slot_id} error: {e}")
                    logger.error(f"Slot {slot_id} error:\n{traceback.format_exc()}")
                    dag.complete_slot(slot_id, False)
    except Exception as e:
        print(f"\n  [!] Execution loop error: {e}")
        logger.error(f"Execution loop error:\n{traceback.format_exc()}")
    
    # Output results
    print("\n" + "="*70)
    print(f" RESULTS: {len(dag.completed)}/{len(framework.slots)} slots completed")
    print("="*70)
    
    # Print memory router stats
    stats = memory_router.get_stats()
    if stats["total_queries"] > 0:
        print(f"\n[Memory Router Stats]")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Memory hits: {stats['memory_hits']} ({stats['memory_hit_rate']*100:.1f}%)")
        print(f"  Expert routes: {stats['expert_routes']}")
    
    for i, slot in enumerate(framework.slots, 1):
        result = memory.get(slot.id)
        print(f"\n--- Section {i}: {slot.title} ---")
        if result:
            print(result.raw_content[:500] + "..." if len(result.raw_content) > 500 else result.raw_content)
            if gui:
                gui.show_section(f"Section {i}: {slot.title}", result.raw_content, result.expert_id)
        else:
            print("[Not completed]")
    
    if gui:
        gui.generation_complete()
        gui.set_status(f"Done! {len(dag.completed)}/{len(framework.slots)} slots")
    
    # === RESET STATE FOR NEXT QUERY ===
    print("\n[Reset] Unloading experts, keeping DeepSeek resident...")
    
    # 1. Unload all experts (except DeepSeek)
    loaded_ids = list(executor.savant_pool.loaded_info.keys())
    for savant_id in loaded_ids:
        if "deepseek" in savant_id.lower():
            continue
        try:
             executor.savant_pool.unload(savant_id)
        except Exception as e:
             logger.warning(f"Error unloading {savant_id}: {e}")

    # 2. Ensure DeepSeek is loaded (for Routing/Supervisor)
    deepseek_path = SAVANT_MODELS.get("security_architect")
    if deepseek_path:
        try:
            executor.savant_pool.load("deepseek_r1", deepseek_path, CUSTOM_VRAM_CONFIG["deepseek_base_mb"])
            print("    ✓ Reset complete (DeepSeek Ready)")
        except Exception as e:
            print(f"    [!] Failed to reload DeepSeek: {e}")

    return len(dag.completed) == len(framework.slots)


def run_headless():
    """Run in headless mode."""
    try:
        success = asyncio.run(run_demo(gui=None))
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_with_gui(demo_mode: bool = False):
    """Run with GUI."""
    
    # Initialize shared components once
    print("\n[Init] Loading shared components...")
    import torch
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_manager = EmbeddingManager(model_name="BAAI/bge-m3", device=embed_device)
    router = DualEmbeddingRouter(embedding_manager, EXPERT_SCOPES)
    scheduler = create_scheduler(profile_name="balanced")
    savant_pool = SavantPool(scheduler_pool=scheduler.savant_pool)
    executor = ModelExecutor(scheduler, savant_pool)
    
    # Stateful components for context retention
    memory = WorkingMemory()
    quality_gate = QualityGate()
    assist_resolver = AssistResolver()
    validator = TemplateValidator()
    
    # Memory Router (Persistent Index)
    memory_router = create_memory_router(
        working_memory=memory,
        vector_router=router,
        embedding_manager=embedding_manager,
        config={"similarity_threshold": QUERY_CONFIG["memory_similarity_threshold"]}
    )
    
    components = {
        "embedding_manager": embedding_manager,
        "router": router,
        "scheduler": scheduler,
        "savant_pool": savant_pool,
        "executor": executor,
        "memory": memory,
        "quality_gate": quality_gate,
        "assist_resolver": assist_resolver,
        "validator": validator,
        "memory_router": memory_router
    }

    def query_handler(query_text: str, gui: CoEDemoGUI):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Pass components to reuse VRAM resources
            loop.run_until_complete(run_demo(gui=gui, query_override=query_text, components=components))
        except Exception as e:
            print(f"[Error] {e}")
            gui.show_progress(f"[ERROR] {e}")
            gui.generation_complete()
        finally:
            loop.close()
    
    gui = CoEDemoGUI(on_query_callback=query_handler, demo_mode=demo_mode)
    
    if demo_mode:
        preset = """Build a personal task manager web app with:
1. A standalone HTML file with embedded CSS and JavaScript
2. SQLite database schema for tasks
3. Python Flask backend with REST API
4. Password hashing with bcrypt"""
        gui.submit_demo_query(preset)
    
    gui.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="College of Experts V13")
    parser.add_argument("--demo", action="store_true", help="Auto-run preset query")
    parser.add_argument("--headless", action="store_true", help="Terminal only")
    args = parser.parse_args()
    
    try:
        if args.headless:
            run_headless()
        else:
            run_with_gui(demo_mode=args.demo)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)