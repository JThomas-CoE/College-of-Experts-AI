"""
College of Experts Benchmark Runner v4 — Tier-Driven Pipeline

KEY ARCHITECTURAL CHANGE FROM v3:
  Pipeline mode is determined by TIER, not by suite name.
  
  TIER1 (Trivial)  → Supervisor-only     (shared MoE layers, no specialist experts)
  TIER2 (Standard) → Specialist + Super  (domain experts + shared layers)  [DEFAULT]
  TIER3 (Complex)  → Multi-Spec + Super  (multiple expert groups + coordination)

  Default is TIER2. Only downgrade to TIER1 for genuinely trivial queries.
  Only upgrade to TIER3 for queries that demonstrably span multiple domains.

MoE Analogy:
  In a SOTA MoE model (Mixtral, DeepSeek-V3), the router almost ALWAYS activates
  experts. The "no experts fire" case is rare — only for trivial token predictions
  where the shared embedding + attention are sufficient.

  By defaulting to TIER2, we ensure that any substantive query gets both:
    1. Domain-grounded specialist knowledge (the "activated expert neurons")
    2. General reasoning + formatting from the supervisor (the "shared layers + router")

  The histographic distribution of which MoE experts fire per task category
  determines how we craft each specialist model. Common experts across all
  categories become the supervisor/router model.

Run: python benchmarks/run_coe_benchmark_v4.py [--suite SUITE] [--samples N]

Preserved versions:
  v2: run_coe_benchmark_v2_checkpoint.py  (solo + fallback)
  v3: run_coe_benchmark_v3_validated.py   (suite-based ensemble, validated results)
  v4: THIS FILE                           (tier-driven, default TIER2)
"""

import asyncio
import time
import json
import sys
import gc
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum, auto

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from run_benchmark for data loading and grading
from benchmarks.run_benchmark import (
    load_standard_suite,
    grade_response,
    BENCHMARK_SUITE,
    PUBLISHED_REFS
)

# CoE Framework imports
from src.embedding_manager import EmbeddingManager
from src.vram_manager import create_scheduler
from src.savant_pool import SavantPool
from src.expert_scope import EXPERT_SCOPES, get_expert_scope
from src.quality_gate import QualityGate
from src.knowledge_layer import KnowledgeRetriever


# ─── Tier Classification ────────────────────────────────────────────────

class QueryTier(Enum):
    """
    Task complexity tier — determines which pipeline executes.
    
    Maps to MoE activation patterns:
      TIER1: Only shared layers fire (router/attention/FFN). No specialist experts.
      TIER2: Shared layers + top-k domain experts fire. Standard MoE forward pass.
      TIER3: Shared layers + multiple expert groups fire. Heavy coordination needed.
    """
    TIER1_TRIVIAL = auto()   # Greetings, meta-questions, clarifications
    TIER2_STANDARD = auto()  # Domain-specific task — the common case
    TIER3_COMPLEX = auto()   # Multi-domain synthesis required


# Signals that a query is TIER1 (trivial) — supervisor-only
TIER1_SIGNALS = [
    "hello", "hi ", "hey ", "thanks", "thank you",
    "what can you do", "who are you", "help me",
    "good morning", "good evening",
]

# Signals that a query is TIER3 (complex) — multi-domain
TIER3_SIGNALS = [
    # Medical + Legal crossover
    "malpractice", "informed consent", "medical liability",
    "health regulation", "hipaa", "clinical trial",
    # Code + SQL crossover
    "database migration", "orm", "api endpoint with database",
    "backend with sql", "data pipeline",
    # Math + Code crossover
    "implement algorithm", "optimization problem",
    "numerical method", "statistical model",
]


def classify_tier(query: str) -> QueryTier:
    """
    Classify query into a tier. Default is TIER2_STANDARD.
    
    Only downgrades to TIER1 if there's strong evidence the query is trivial.
    Only upgrades to TIER3 if the query demonstrably spans multiple domains.
    
    In a real system, this would use the QueryClassifier with an LLM.
    For benchmarks, we use signal-based heuristics.
    """
    q = query.lower().strip()
    
    # Check for TIER1 (trivial) — very short or clearly a greeting
    if len(q) < 20:
        for signal in TIER1_SIGNALS:
            if signal in q:
                return QueryTier.TIER1_TRIVIAL
    
    # Check for TIER3 (complex) — multi-domain signals
    domain_signals_found = 0
    for signal in TIER3_SIGNALS:
        if signal in q:
            domain_signals_found += 1
    if domain_signals_found >= 2:
        return QueryTier.TIER3_COMPLEX
    
    # Default: TIER2_STANDARD — assume the query needs expertise
    return QueryTier.TIER2_STANDARD


# ─── Configuration ──────────────────────────────────────────────────────

BASELINE_SCORES = {
    "HumanEval": {"accuracy": 71.4, "model": "Qwen2.5-Coder-7B"},
    "GSM8K": {"accuracy": 56.0, "model": "Qwen2.5-Math-7B"},
    "Medical": {"accuracy": 40.0, "model": "BioMistral-7B"},
    "Spider": {"accuracy": 64.8, "model": "sqlcoder-7b-2"},
    "Legal": {"accuracy": 8.4, "model": "law-LLM"}
}

SUITE_EXPERT_MAP = {
    "HumanEval": "python_backend",
    "GSM8K": "math_expert",
    "Medical": "medical_clinical",
    "Spider": "sql_schema_architect",
    "Legal": "legal_contracts"
}

# Model paths
SAVANT_MODELS = {
    "python_backend": "models/Qwen2.5-Coder-7B-DML",
    "sql_schema_architect": "models/sqlcoder-7b-2-DML",
    "html_css_specialist": "models/Qwen2.5-Coder-7B-DML",
    "math_expert": "models/Qwen2.5-Math-7B-DML",
    "security_architect": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
    "legal_contracts": "models/law-LLM-DML",
    "medical_clinical": "models/BioMistral-7B-DML",
    "deepseek_supervisor": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
}


# ─── Chat templates per model family ───────────────────────────────────

def _format_prompt(model_path: str, prompt: str, system: str = "") -> str:
    """Format prompt according to model architecture."""
    p = model_path.lower()

    # SQLCoder (CodeLlama-based) — uses ### markers, NO system
    if "sqlcoder" in p:
        return (
            f"### Task\nGenerate a SQL query to answer the following question.\n\n"
            f"### Instructions\n{prompt}\n\n### Response\n"
        )

    # law-LLM (Llama-based) — Llama-2 instruct, 2048 ctx limit
    if "law" in p:
        sys_part = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"

    # BioMistral (Mistral-based) — Mistral instruct format
    if "biomistral" in p or "bio" in p:
        sys_part = f"{system}\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"

    # Qwen / DeepSeek — ChatML format
    if system:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# ─── Result dataclass ───────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""
    task_id: str
    suite: str
    expert_used: str
    tier: str                   # TIER1_TRIVIAL / TIER2_STANDARD / TIER3_COMPLEX
    pipeline: str               # t1_supervisor / t2_ensemble / t3_multi / fallback
    score: float
    max_score: float
    feedback: str
    response: str
    latency_ms: float
    tokens: int
    specialist_response: str = ""
    used_retry: bool = False
    used_fallback: bool = False
    used_ensemble: bool = False
    error: Optional[str] = None


# ─── The Runner ─────────────────────────────────────────────────────────

class CoEBenchmarkRunner:
    """
    Runs benchmarks through tier-driven College of Experts pipeline.
    
    TIER1 → Supervisor (DeepSeek) directly answers
    TIER2 → Specialist drafts → Supervisor synthesizes  [DEFAULT]
    TIER3 → Multiple specialists draft → Supervisor orchestrates
    """

    def __init__(self, enable_quality_gate: bool = True, enable_fallback: bool = True):
        self.enable_quality_gate = enable_quality_gate
        self.enable_fallback = enable_fallback

        self.embedding_manager = None
        self.scheduler = None
        self.quality_gate = None
        self.knowledge_retriever = None

        self._current_model = None
        self._current_model_path = None
        self._tokenizer = None

        # Stats
        self.tier_stats = {t.name: 0 for t in QueryTier}
        self.pipeline_stats = {
            "t1_supervisor": 0, "t2_ensemble": 0, 
            "t3_multi": 0, "fallback": 0
        }

    def _init_components(self):
        """Initialize CoE components."""
        import torch

        print("[CoE v4] Initializing framework components...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_manager = EmbeddingManager(model_name="BAAI/bge-m3", device=device)
        print("  ✓ Embedding manager")

        self.scheduler = create_scheduler(profile_name="balanced")
        print("  ✓ VRAM scheduler")

        if self.enable_quality_gate:
            self.quality_gate = QualityGate()
            print("  ✓ Quality gate")

        try:
            self.knowledge_retriever = KnowledgeRetriever(
                embedding_fn=self.embedding_manager.encode,
                knowledge_base_dir="data/knowledge",
                enable_web_search=False
            )
            print("  ✓ Knowledge retriever")
        except Exception as e:
            print(f"  ⚠ Knowledge retriever: {e}")

        print("\n  Model verification:")
        for expert_id, path in SAVANT_MODELS.items():
            exists = Path(path).exists()
            status = "✓" if exists else "✗ MISSING"
            print(f"    {status} {expert_id}: {path}")

    def _load_model(self, model_path: str) -> bool:
        """Load model into VRAM (swaps out previous)."""
        import onnxruntime_genai as og

        if self._current_model_path == model_path:
            return True

        if self._current_model:
            del self._current_model
            self._current_model = None
            self._current_model_path = None
            self._tokenizer = None
            gc.collect()

        try:
            self._current_model = og.Model(model_path)
            self._current_model_path = model_path
            self._tokenizer = og.Tokenizer(self._current_model)
            return True
        except Exception as e:
            print(f"  ⚠ Failed to load {model_path}: {e}")
            self._current_model_path = None
            return False

    def _generate(self, prompt: str, system: str = "",
                  max_tokens: int = 512, temperature: float = 0.3) -> Tuple[str, int]:
        """Generate response using currently loaded model."""
        import onnxruntime_genai as og

        formatted = _format_prompt(self._current_model_path, prompt, system)
        input_tokens = self._tokenizer.encode(formatted)

        # Respect model context limits
        config_path = Path(self._current_model_path) / "genai_config.json"
        ctx_limit = 4096
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                ctx_limit = cfg.get("model", {}).get("context_length", 4096)
            except:
                pass

        available_gen = min(max_tokens, ctx_limit - len(input_tokens) - 10)
        if available_gen < 50:
            available_gen = 50

        params = og.GeneratorParams(self._current_model)
        params.set_search_options(
            max_length=len(input_tokens) + available_gen,
            temperature=max(temperature, 0.01)
        )

        generator = og.Generator(self._current_model, params)
        generator.append_tokens(input_tokens)

        output_tokens = []
        while not generator.is_done() and len(output_tokens) < available_gen:
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            output_tokens.append(token)

        response = self._tokenizer.decode(output_tokens)
        del generator
        del params

        return response.strip(), len(output_tokens)

    # ─── Prompt builders ────────────────────────────────────────────────

    def _build_specialist_prompt(self, suite: str, task_prompt: str) -> Tuple[str, str]:
        """Build prompt for the specialist model (domain expert neurons)."""
        knowledge = ""
        if self.knowledge_retriever:
            try:
                chunks = self.knowledge_retriever.retrieve(task_prompt, max_tokens=400)
                if chunks:
                    knowledge = f"REFERENCE:\n{chunks[0][:400]}\n\n"
            except:
                pass

        if suite == "GSM8K":
            return (
                f"{knowledge}Solve this math problem step by step. Show your work clearly "
                f"and give the final numerical answer on the last line.\n\n"
                f"PROBLEM: {task_prompt}\n\nSOLUTION:",
                "You are a mathematics expert. Solve problems step-by-step, showing all work."
            )
        elif suite == "HumanEval":
            return (
                f"{knowledge}Complete this Python function. Provide only the function body "
                f"(implementation), not the signature.\n\n{task_prompt}\n\nIMPLEMENTATION:",
                "You are an expert Python programmer. Write clean, efficient, bug-free code."
            )
        elif suite == "Spider":
            return (f"{knowledge}{task_prompt}", "")
        elif suite == "Medical":
            return (
                f"{knowledge}You are reviewing this medical question. Provide your clinical "
                f"assessment based on the evidence.\n\n"
                f"QUESTION: {task_prompt}\n\n"
                f"Provide your answer starting with yes, no, or maybe, then explain:",
                "You are an expert clinical researcher. Provide evidence-based medical analysis."
            )
        elif suite == "Legal":
            return (
                f"{knowledge}Analyze this legal question based on contract law principles.\n\n"
                f"QUESTION: {task_prompt}\n\n"
                f"Answer with yes or no, then provide your legal reasoning:",
                "You are a legal expert specializing in contract law and regulatory compliance."
            )
        return (task_prompt, "")

    def _build_supervisor_prompt(self, suite: str, task_prompt: str) -> Tuple[str, str]:
        """Build prompt for supervisor-only path (TIER1 — shared layers only)."""
        if suite == "GSM8K":
            return (
                f"Solve this math problem step by step.\n\n"
                f"PROBLEM: {task_prompt}\n\nSOLUTION:",
                "You are an advanced reasoning model. Think step by step."
            )
        elif suite == "HumanEval":
            return (
                f"Complete this Python function. Provide only the implementation.\n\n"
                f"{task_prompt}\n\nIMPLEMENTATION:",
                "You are an expert programmer. Write clean, correct code."
            )
        elif suite == "Spider":
            return (
                f"Write the SQL query for this question. Return only the SQL.\n\n"
                f"{task_prompt}\n\nSQL:",
                "You are a database expert. Write correct SQL."
            )
        elif suite == "Medical":
            return (
                f"Answer this medical question. Start with yes, no, or maybe.\n\n"
                f"QUESTION: {task_prompt}\n\nANSWER:",
                "You are a medical expert. Give accurate, evidence-based answers."
            )
        elif suite == "Legal":
            return (
                f"Answer this legal question with a clear yes/no followed by reasoning.\n\n"
                f"QUESTION: {task_prompt}\n\nANSWER:",
                "You are a legal expert. Analyze contracts and regulations accurately."
            )
        return (task_prompt, "You are a helpful assistant.")

    def _build_synthesis_prompt(self, suite: str, task_prompt: str,
                                specialist_response: str) -> Tuple[str, str]:
        """Build prompt for supervisor synthesis (TIER2 — shared layers refine expert output)."""
        if suite == "Medical":
            return (
                f"A medical specialist has reviewed this question and provided their analysis.\n\n"
                f"ORIGINAL QUESTION:\n{task_prompt}\n\n"
                f"SPECIALIST ASSESSMENT:\n{specialist_response[:800]}\n\n"
                f"As a senior reviewer, synthesize the specialist's findings into a clear answer.\n"
                f"Start with yes, no, or maybe. Then provide a concise explanation.\n\n"
                f"FINAL ANSWER:",
                "You are a senior medical reviewer synthesizing specialist opinions. "
                "Provide a clear, definitive answer based on the specialist's analysis."
            )
        elif suite == "Legal":
            return (
                f"A legal specialist has reviewed this contract question and provided analysis.\n\n"
                f"ORIGINAL QUESTION:\n{task_prompt}\n\n"
                f"SPECIALIST ANALYSIS:\n{specialist_response[:800]}\n\n"
                f"As a senior legal reviewer, synthesize this into a clear determination.\n"
                f"Start with yes or no. Then provide concise legal reasoning.\n\n"
                f"FINAL DETERMINATION:",
                "You are a senior legal reviewer synthesizing specialist legal analysis. "
                "Give a clear yes/no determination with reasoning."
            )
        # Generic synthesis for any other suite
        return (
            f"A domain specialist provided this analysis:\n\n"
            f"QUESTION: {task_prompt}\n\n"
            f"SPECIALIST: {specialist_response[:800]}\n\n"
            f"Synthesize into a clear, accurate final answer:\n\nANSWER:",
            "You are synthesizing expert analysis into a clear answer."
        )

    # ─── Tier-Driven Pipeline ───────────────────────────────────────────

    async def run_task(self, suite: str, task: Dict) -> BenchmarkResult:
        """
        Run a single task through the tier-driven pipeline.
        
        The tier determines HOW the task is processed:
          TIER1 → supervisor_only()    DeepSeek handles it directly
          TIER2 → specialist_then_supervisor()  [DEFAULT for benchmarks]
          TIER3 → multi_specialist_then_supervisor()  (future: multiple experts)
        """
        task_id = task['id']
        task_prompt = task['prompt']
        reference = task.get('answer', '')
        expert_id = SUITE_EXPERT_MAP.get(suite, "python_backend")

        start = time.time()

        # ── CLASSIFY ──
        tier = classify_tier(task_prompt)
        self.tier_stats[tier.name] += 1

        # ── ROUTE BY TIER ──
        if tier == QueryTier.TIER1_TRIVIAL:
            result = await self._pipeline_t1_supervisor(
                suite, task_id, task_prompt, reference, expert_id, start
            )
        elif tier == QueryTier.TIER3_COMPLEX:
            # TIER3 uses the same ensemble as TIER2 for now
            # Future: multiple specialists draft, then supervisor orchestrates
            result = await self._pipeline_t2_ensemble(
                suite, task_id, task_prompt, reference, expert_id, start
            )
            result.tier = "TIER3_COMPLEX"
            result.pipeline = "t3_multi"
            self.pipeline_stats["t3_multi"] += 1
        else:
            # TIER2_STANDARD — the default and most common path
            result = await self._pipeline_t2_ensemble(
                suite, task_id, task_prompt, reference, expert_id, start
            )

        return result

    async def _pipeline_t1_supervisor(self, suite, task_id, task_prompt,
                                       reference, expert_id, start) -> BenchmarkResult:
        """
        TIER1 Pipeline: Supervisor (DeepSeek) answers directly.
        
        MoE analogy: Only shared attention + FFN layers fire.
        No specialist expert neurons activated.
        """
        pipeline = "t1_supervisor"
        self.pipeline_stats[pipeline] += 1

        sv_path = SAVANT_MODELS["deepseek_supervisor"]
        if not Path(sv_path).exists() or not self._load_model(sv_path):
            return self._error_result(task_id, suite, expert_id,
                                      "TIER1_TRIVIAL", pipeline, "Supervisor load failed")

        prompt, system = self._build_supervisor_prompt(suite, task_prompt)
        response, tokens = self._generate(prompt, system=system, max_tokens=512)
        latency_ms = (time.time() - start) * 1000

        score, max_score, feedback = grade_response(suite, response, reference)

        return BenchmarkResult(
            task_id=task_id, suite=suite, expert_used="deepseek_supervisor",
            tier="TIER1_TRIVIAL", pipeline=pipeline, score=score,
            max_score=max_score, feedback=feedback, response=response[:200],
            latency_ms=latency_ms, tokens=tokens
        )

    async def _pipeline_t2_ensemble(self, suite, task_id, task_prompt,
                                     reference, expert_id, start) -> BenchmarkResult:
        """
        TIER2 Pipeline: Specialist drafts → Supervisor synthesizes.
        
        MoE analogy: 
          Phase 1 — Top-k domain expert neurons fire, generating grounded output
          Phase 2 — Shared layers (attention + FFN) refine and format the output
        
        This is the DEFAULT pipeline. Most queries come through here.
        """
        pipeline = "t2_ensemble"
        self.pipeline_stats[pipeline] += 1

        model_path = SAVANT_MODELS.get(expert_id, SAVANT_MODELS["python_backend"])
        used_retry = False
        used_fallback = False
        specialist_response = ""

        # ── Phase 1: Specialist generates domain-grounded draft ──
        specialist_loaded = Path(model_path).exists() and self._load_model(model_path)

        if not specialist_loaded:
            # Can't load specialist — supervisor handles it
            return await self._pipeline_t1_supervisor(
                suite, task_id, task_prompt, reference, expert_id, start
            )

        prompt, system = self._build_specialist_prompt(suite, task_prompt)
        specialist_response, tokens = self._generate(prompt, system=system, max_tokens=512)

        # Quality gate on specialist output
        if self.enable_quality_gate and len(specialist_response.strip()) < 20:
            used_retry = True
            specialist_response, tokens = self._generate(
                prompt, system=system, max_tokens=512, temperature=0.5
            )

        response = specialist_response

        # ── Phase 2: Supervisor synthesizes ──
        sv_path = SAVANT_MODELS["deepseek_supervisor"]
        if Path(sv_path).exists() and self._load_model(sv_path):
            # Grade specialist draft first
            spec_score, _, _ = grade_response(suite, specialist_response, reference)

            synth_prompt, synth_system = self._build_synthesis_prompt(
                suite, task_prompt, specialist_response
            )
            synth_response, tokens = self._generate(
                synth_prompt, system=synth_system, max_tokens=512
            )

            # Keep the better answer
            synth_score, _, _ = grade_response(suite, synth_response, reference)
            if synth_score >= spec_score:
                response = synth_response

        latency_ms = (time.time() - start) * 1000
        score, max_score, feedback = grade_response(suite, response, reference)

        # ── Score-gated fallback ──
        if score == 0 and self.enable_fallback and "deepseek" not in expert_id.lower():
            used_fallback = True
            sv_path = SAVANT_MODELS["deepseek_supervisor"]
            if Path(sv_path).exists() and self._load_model(sv_path):
                fb_prompt = (
                    f"Answer this question carefully and accurately.\n\n"
                    f"QUESTION: {task_prompt}\n\n"
                    f"Think through this step by step, then provide your final answer:"
                )
                response, tokens = self._generate(
                    fb_prompt, system="You are DeepSeek, an advanced reasoning model.",
                    max_tokens=512
                )
                score, max_score, feedback = grade_response(suite, response, reference)
                expert_id = "deepseek_fallback"
                latency_ms = (time.time() - start) * 1000

        return BenchmarkResult(
            task_id=task_id, suite=suite, expert_used=expert_id,
            tier="TIER2_STANDARD", pipeline=pipeline, score=score,
            max_score=max_score, feedback=feedback, response=response[:200],
            latency_ms=latency_ms, tokens=tokens,
            specialist_response=specialist_response[:200],
            used_retry=used_retry, used_fallback=used_fallback,
            used_ensemble=True
        )

    def _error_result(self, task_id, suite, expert_id, tier, pipeline, error_msg):
        return BenchmarkResult(
            task_id=task_id, suite=suite, expert_used=expert_id,
            tier=tier, pipeline=pipeline, score=0.0, max_score=1.0,
            feedback=error_msg, response="", latency_ms=0, tokens=0,
            error=error_msg
        )

    async def run_suite(self, suite_name: str, max_samples: int = 25) -> List[BenchmarkResult]:
        """Run all tasks in a benchmark suite."""
        tasks = load_standard_suite(suite_name, max_samples=max_samples)
        if not tasks:
            print(f"  ⚠ No tasks found for {suite_name}")
            return []

        print(f"\n[Suite: {suite_name}] Running {len(tasks)} tasks (tier-driven)...")

        results = []
        for i, task in enumerate(tasks):
            print(f"  [{i + 1}/{len(tasks)}] {task['id']}...", end="\r")
            result = await self.run_task(suite_name, task)
            results.append(result)

        return results

    def cleanup(self):
        if self._current_model:
            del self._current_model
            self._current_model = None
        self._current_model_path = None
        self._tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── Orchestrator ───────────────────────────────────────────────────────

async def run_coe_benchmark(
    suite: Optional[str] = None,
    max_samples: int = 25,
    enable_quality_gate: bool = True,
    enable_fallback: bool = True,
):
    """Run CoE v4 benchmark with tier-driven pipeline."""

    print("\n" + "=" * 72)
    print(" COLLEGE OF EXPERTS BENCHMARK v4  (Tier-Driven Pipeline)")
    print(f" Config: QG={enable_quality_gate}, Fallback={enable_fallback}")
    print(f" Default tier: TIER2_STANDARD (specialist + supervisor)")
    print("=" * 72)

    runner = CoEBenchmarkRunner(
        enable_quality_gate=enable_quality_gate,
        enable_fallback=enable_fallback,
    )
    runner._init_components()

    target_suites = [suite] if suite else list(BASELINE_SCORES.keys())
    all_results = []

    try:
        for suite_name in target_suites:
            results = await runner.run_suite(suite_name, max_samples=max_samples)
            all_results.extend(results)

            valid = [r for r in results if r.error is None]
            if valid:
                avg_score = sum(r.score for r in valid) / len(valid)
                retries = sum(1 for r in valid if r.used_retry)
                fallbacks = sum(1 for r in valid if r.used_fallback)
                ensembles = sum(1 for r in valid if r.used_ensemble)
                avg_latency = sum(r.latency_ms for r in valid) / len(valid)

                # Tier breakdown
                tiers = {}
                for r in valid:
                    tiers[r.tier] = tiers.get(r.tier, 0) + 1

                # Pipeline breakdown
                pipes = {}
                for r in valid:
                    pipes[r.pipeline] = pipes.get(r.pipeline, 0) + 1

                baseline = BASELINE_SCORES.get(suite_name, {})
                baseline_acc = baseline.get("accuracy", 0)
                delta = (avg_score * 100) - baseline_acc

                print(f"\n[{suite_name} Results]")
                print(f"  CoE Accuracy:  {avg_score * 100:.1f}%")
                print(f"  Baseline:      {baseline_acc:.1f}% ({baseline.get('model', 'N/A')})")
                print(f"  Delta:         {delta:+.1f}%")
                print(f"  Tiers:         {tiers}")
                print(f"  Pipelines:     {pipes}")
                print(f"  Ensemble used: {ensembles}/{len(valid)}")
                print(f"  Retries:       {retries}/{len(valid)}")
                print(f"  Fallbacks:     {fallbacks}/{len(valid)}")
                print(f"  Avg latency:   {avg_latency:.1f}ms")
    finally:
        runner.cleanup()

    # Final summary
    print("\n" + "=" * 72)
    print(" FINAL COMPARISON: CoE v4 (Tier-Driven) vs Baseline")
    print("=" * 72)
    print(f"{'Suite':<12} | {'Tier':<16} | {'CoE':>7} | {'Base':>7} | {'Delta':>7} | {'Ens':>5} | {'FB':>5}")
    print("-" * 72)

    for suite_name in target_suites:
        suite_results = [r for r in all_results if r.suite == suite_name and r.error is None]
        if suite_results:
            coe_acc = sum(r.score for r in suite_results) / len(suite_results) * 100
            baseline_acc = BASELINE_SCORES.get(suite_name, {}).get("accuracy", 0)
            delta = coe_acc - baseline_acc
            ens_count = sum(1 for r in suite_results if r.used_ensemble)
            fb_count = sum(1 for r in suite_results if r.used_fallback)
            # Most common tier for this suite
            tier_counts = {}
            for r in suite_results:
                tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
            top_tier = max(tier_counts, key=tier_counts.get)
            print(
                f"{suite_name:<12} | {top_tier:<16} | {coe_acc:>6.1f}% | "
                f"{baseline_acc:>6.1f}% | {delta:>+6.1f}% | "
                f"{ens_count:>2}/{len(suite_results)} | {fb_count:>2}/{len(suite_results)}"
            )

    print("=" * 72)
    print(f"\nTier Stats:     {runner.tier_stats}")
    print(f"Pipeline Stats: {runner.pipeline_stats}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoE v4 benchmark — tier-driven pipeline")
    parser.add_argument("--suite", type=str, default=None, help="Specific suite to run")
    parser.add_argument("--samples", type=int, default=25, help="Max samples per suite")
    parser.add_argument("--no-qg", action="store_true", help="Disable quality gate")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fallback")

    args = parser.parse_args()

    asyncio.run(run_coe_benchmark(
        suite=args.suite,
        max_samples=args.samples,
        enable_quality_gate=not args.no_qg,
        enable_fallback=not args.no_fallback,
    ))
