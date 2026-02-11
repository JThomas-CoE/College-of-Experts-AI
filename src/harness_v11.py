
import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import time
import json
import traceback
import threading
import gc
try:
    import torch
except ImportError:
    torch = None

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from dataclasses import dataclass, field

from .crosstalk_v8 import CrosstalkBus
from .expert_slots_v8 import ExpertSlotManager
from .router_v9 import TaskChunk 
from .semantic_router import SemanticRouter

class HarnessV11:
    """
    V11 'Semantic-Deep Reasoning' Harness.
    - Planner: DeepSeek-R1-Distill (Slot 0) - Permanent Resident
    - Router: BAAI/bge-m3 Vectors
    - Savants: GPU Specialists (Slots 1-3)
    """
    
    DECOMPOSE_PROMPT = """Role: Chief Technical Architect
    Task: Decompose the request into EXACTLY 3 ATOMIC chunks.
    
    ARCHITECTURE: The Triangle Theory.
    1. Technical/Scripting
    2. Regulatory/Policy
    3. Clinical/Operational
    
    Output JSON:
    {{ "chunks": [ {{ "id": "c1", "description": "...", "assigned_expert": "..." }} ] }}
    
    Query: {query}
    """
    
    SHARPEN_PLAN_PROMPT = """Role: Chief Technical Architect
    Task: These tasks are 'Semantically Muddy'. They hit multiple experts.
    
    MUDDY LIST:
    {muddy_report}
    
    ACTION: 
    1. Maintain exactly 3 chunks.
    2. Rewrite the descriptions to be SHARPLY distinct.
    3. Use 'Code'/'Script' for the Python Expert. Use 'Cite'/'Statute' for the Legal Expert.
    
    Output JSON Plan:
    {{ "chunks": [ ... ] }}
    """

    OUTLINE_PROMPT = """You are the Lead Solutions Architect. Analyze the user's request and outline the KEY FUNCTIONAL COMPONENTS required.
    
    User Request: {query}
    
    Task:
    1. Identify the distinct domains (e.g. Code vs Legal vs Security).
    2. List the high-level steps needed to solve this.
    3. Architecture: How do these components fit together?
    
    Output a concise text outline. Do not output JSON yet."""

    DECOMPOSE_PROMPT = """Based on the Architectural Plan below, break the work into DISTINCT implementation chunks.
    
    Plan:
    {plan}
    
    CRITICAL: Partition tasks to MAXIMIZE DIFFERENTIATION. Avoid overlapping concerns.
    - Python Expert -> Code only.
    - Legal Expert -> Compliance/Text only.
    - Security Expert -> Audit/Validation only.
    
    Respond in JSON format:
    {{
        "chunks": [
            {{ "id": "c1", "description": "...", "assigned_expert": "...", "required_capabilities": [...] }}
        ]
    }}"""

    SYNTHESIS_SYSTEM_PROMPT = "You are an expert synthesist. Integrate these specialist outputs into a single, cohesive, professional report for the user."

    def __init__(self, config: Any):
        self.config = config
        self.catalog = config.catalog
        self.initialized = False
        self.slot_manager = ExpertSlotManager(catalog=self.catalog, num_slots=8)
        self.router = SemanticRouter(self.catalog)
        self.bus = CrosstalkBus()
        self.supervisor_model = None

    def initialize(self):
        if self.initialized: return
        print("[Harness V11] Pining DeepSeek Supervisor to Slot 0...")
        # load_expert allows explicit slot assignment (Slot 0 for Supervisor residency)
        self.supervisor_model = self.slot_manager.load_expert("deepseek_reasoner", slot_id=0)
        self.initialized = True

    def _sanitize_output(self, text: str) -> str:
        """Strip dangerous control tokens that cause model collapse."""
        if not text: return ""
        # Remove ChatML tokens
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        # Remove other common leakage
        text = text.replace("<s>", "").replace("</s>", "")
        return text.strip()

    def process(self, user_input: str) -> str:
        self.initialize()
        
        # --- PHASE 2: PLANNING & SEMANTIC SHARPENING ---
        # This occurs entirely in Slot 0. No Savants are loaded yet.
        print("\n[V11 Phase 2] Planning & Semantic Sharpening...")
        chunks = self._decompose_and_sharpen(user_input)
        
        # --- PHASE 3: GLOBAL TEAM OPTIMIZATION ---
        print("\n[V11 Phase 3] Global Team Optimization...")
        chunk_dicts = [{"id": c.id, "description": c.description, "assigned_expert": c.assigned_expert} for c in chunks]
        optimized = self.router.optimize_team_assignment(chunk_dicts)
        
        # Robust Rehydration (Ensures expert_id is a string, not a list)
        final_plan = []
        for d in optimized:
            exp = d["assigned_expert"]
            if isinstance(exp, list) and exp: exp = exp[0]
            
            final_plan.append(TaskChunk(
                id=d["id"], 
                description=d["description"], 
                assigned_expert=str(exp), 
                required_capabilities=[]
            ))

        # --- PHASE 4: EXECUTION (VRAM SAFE) ---
        print("\n[V11 Phase 4] Executing Triangle of Experts (Parallel)...")
        results = self._execute_parallel(final_plan)
        
        # SANITIZE before Optimization
        for r in results:
            r["content"] = self._sanitize_output(r["content"])
        
        # --- PHASE 4.5: Global Context Optimization (Holistic VRAM Safety) ---
        results = self._optimize_context_budget(results)
        
        # SANITIZE AGAIN (Safety)
        for r in results:
            r["content"] = self._sanitize_output(r["content"])
        
        # --- PHASE 5: RESIDENT SYNTHESIS ---
        print("\n[V11 Phase 5] Resident Synthesis (Zero-Reload)...")
        return self._synthesize(user_input, results)

    def _decompose_and_sharpen(self, query: str, max_rounds: int = 2) -> List[TaskChunk]:
        # Step 1: Architectural Outline
        print("[V11 Phase 2] Generating Architectural Outline...")
        outline_prompt = [{"role": "user", "content": self.OUTLINE_PROMPT.format(query=query)}]
        outline = self._run_inference(self.supervisor_model, outline_prompt, max_tokens=1024)
        print(f"  [Plan] {outline[:100]}...")
        
        # Step 2: Differentiated Partitioning
        print("[V11 Phase 2] Partitioning Tasks for Maximum Differentiation...")
        decompose_prompt_text = self.DECOMPOSE_PROMPT.format(plan=outline)
        decompose_prompt = [{"role": "user", "content": decompose_prompt_text}]
        
        response = self._run_inference(self.supervisor_model, decompose_prompt, max_tokens=2048)
        
        # Parse JSON
        start = response.find("```json")
        if start != -1:
            response = response[start+7:]
            end = response.find("```")
            if end != -1: response = response[:end]
            
        try:
            data = json.loads(response.strip())
            chunks = []
            for d in data.get("chunks", []):
                # Map expert name to ID
                exp = d.get("assigned_expert", "python_expert")
                if "python" in exp.lower(): exp = "python_expert"
                elif "legal" in exp.lower(): exp = "legal_expert"
                elif "security" in exp.lower(): exp = "security_expert"
                elif "medical" in exp.lower(): exp = "medical_expert"
                
                chunks.append(TaskChunk(
                    id=d.get("id", f"c{len(chunks)+1}"),
                    description=d["description"], 
                    assigned_expert=str(exp), 
                    required_capabilities=d.get("required_capabilities", [])
                ))
            # The sharpening loop below expects 3 chunks. If we don't get 3, we might need to adjust or fallback.
            # For now, let's ensure we have at least one chunk to avoid errors in the sharpening loop.
            if not chunks:
                print("[V11] Decompose returned no chunks. Falling back to single chunk.")
                chunks = [TaskChunk(id="c1", description=query, assigned_expert="python_expert", required_capabilities=[])]
            
        except Exception as e:
            print(f"[V11] Decompose Failed: {e}. Fallback to monolith.")
            chunks = [TaskChunk(id="c1", description=query, assigned_expert="python_expert", required_capabilities=[])]
        
        # 2. Sharpening Loop (Quality Gate) - This part remains from the original method
        for rd in range(max_rounds):
            chunk_dicts = [{"id": c.id, "description": c.description} for c in chunks]
            reports = self.router.analyze_plan_clarity(chunk_dicts)
            
            muddy = [r for r in reports if r['is_muddy']]
            if not muddy:
                print(f"  -> Plan is Sharp (Round {rd+1}).")
                break
                
            print(f"  -> Round {rd+1}: {len(muddy)} muddy tasks found. Sharpening...")
            muddy_lines = [f"- {r['id']}: '{r['description']}' matches {r['top_expert']} & {r['runner_up']}." for r in muddy]
            sharpen_prompt = self.SHARPEN_PLAN_PROMPT.format(muddy_report="\n".join(muddy_lines))
            
            new_chunks = self._parse_chunks(self._query_supervisor(sharpen_prompt))
            if new_chunks and len(new_chunks) == 3:
                chunks = new_chunks
            else:
                print(f"  -> WARNING: Sharpening rejected (Count drift). Keeping previous.")
                
        return chunks

    _load_lock = threading.Lock() # Serializes hardware allocation to prevent peak spikes

    def _execute_parallel(self, chunks: List[TaskChunk]) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                # We limit our specialists to Slots 1, 2, and 3
                slot_id = (i % 3) + 1
                futures.append(executor.submit(self._execute_single_task, chunk, slot_id))
            
            for f in as_completed(futures):
                results.append(f.result())
        return results

    def _execute_single_task(self, chunk: TaskChunk, slot_id: int) -> Dict:
        # 1. Serialized Hardware Prep (The VRAM Guard)
        with self._load_lock:
            print(f"  [Slot {slot_id}] Preparing hardware for {chunk.assigned_expert}...")
            # load_expert allows explicit hardware slot assignment
            instance = self.slot_manager.load_expert(chunk.assigned_expert, slot_id=slot_id)
            time.sleep(1.0) # Settle DirectML context

        # 2. Input Sanitization (Fixes Code Gen)
        task_text = chunk.description
        if isinstance(task_text, dict):
            task_text = task_text.get("task") or task_text.get("description") or str(task_text)

        # 3. Inference
        print(f"  > [Slot {slot_id}] Generating output for {chunk.id}...")
        messages = [
            {"role": "system", "content": instance.expert_def.system_prompt},
            {"role": "user", "content": task_text}
        ]
        
        try:
            # Safe limit: 1280 tokens leave room for ~700 tokens of input prompt within 2048 limit
            content = self._run_inference(instance, messages, max_tokens=1280)
            
            # --- CHECKPOINT (Quality Only - Length Managed Globally) ---
            qa_persona = "You are a Quality Assurance Specialist. Review the work submitted by an expert."
            
            # Augmented Checkpoint Prompt (Focus on Quality/Safety)
            checkpoint_prompt_text = f"""Original Task: {chunk.description}
Review the Expert's Output below.
    
Evaluation Criteria:
1. Did they follow instructions?
2. Is the content valid/safe?
(Length is managed globally, ignore conciseness unless verbose)

Expert Output:
{content[:2000]}...

Respond in JSON: {{ "approved": boolean, "feedback": "string" }}"""

            checkpoint_messages = [
                {"role": "system", "content": qa_persona},
                {"role": "user", "content": checkpoint_prompt_text}
            ]
            
            # Run Checkpoint (Same Slot)
            qa_response = self._run_inference(instance, checkpoint_messages, max_tokens=512)
            qa_data = self._parse_json(qa_response)
            
            if qa_data and not qa_data.get("approved", True):
                print(f"  [V11] Chunk {chunk.id} Rejected: {qa_data.get('feedback')}")
                # 2. Refine (Switch back to Expert Persona + Fresh Context)
                # 2. Refine (Hybrid Strategy: Savant First, Supervisor Fallback)
                refine_prompt = [
                    {"role": "system", "content": f"You are acting as the {chunk.assigned_expert}. {instance.expert_def.system_prompt}"},
                    {"role": "user", "content": f"Task: {chunk.description}\n\nDraft Output:\n{content}\n\nQA Feedback: {qa_data.get('feedback')}\n\nRewrite the output to address feedback. IMPORTANT: Be as concise as possible while maintaining correctness."}
                ]
                
                # Estimate Tokens (Char/4 approx)
                est_input_tokens = len(str(refine_prompt)) // 4
                
                # Dynamic Limits based on Expert Architecture
                is_code = "python" in str(chunk.assigned_expert).lower()
                SAFE_CTX_LIMIT = 8192 if is_code else 2048
                max_gen_tokens = 4096 if is_code else 1500
                
                # Check against limit (leaving room for generation)
                if est_input_tokens + 512 > SAFE_CTX_LIMIT:
                    print(f"  [V11] Refinement Context ({est_input_tokens} toks) > Limit ({SAFE_CTX_LIMIT}). Escalating to Supervisor.")
                    exec_model = self.supervisor_model
                    # Supervisor (DeepSeek) handles larger generation easily
                    max_gen_tokens = 4096 
                else:
                    exec_model = instance
                
                content = self._run_inference(exec_model, refine_prompt, max_tokens=max_gen_tokens)
                print(f"  [V11] Chunk {chunk.id} Refined.")
            
            print(f"  [OK] Chunk {chunk.id} finished.")
            return {"id": chunk.id, "expert": chunk.assigned_expert, "content": content, "slot_id": slot_id}
        except Exception as e:
            print(f"  [FAIL] Chunk {chunk.id}: {e}")
            return {"id": chunk.id, "expert": chunk.assigned_expert, "content": f"Failed: {e}", "slot_id": slot_id}

    def _optimize_context_budget(self, results: List[Dict]):
        """
        Phase 4.5: Holistically optimize context size based on total budget.
        Re-prompts resident experts to compress if total > 10,000 chars.
        """
        MAX_TOTAL_CHARS = 10000
        
        while True:
            total_chars = sum(len(r["content"]) for r in results)
            if total_chars <= MAX_TOTAL_CHARS:
                print(f"  [V11] Context Budget Safe: {total_chars}/{MAX_TOTAL_CHARS} chars.")
                break
                
            print(f"  [V11] Context Overflow ({total_chars} chars). Optimizing...")
            
            # Identify candidate to compress (Largest Non-Code chunk first)
            # We assume 'python' experts produce code that shouldn't be squeezed too hard.
            candidates = sorted(results, key=lambda x: len(x["content"]), reverse=True)
            target = candidates[0]
            
            # If largest is Python and decent size, try next largest? 
            # (Simple heuristic: Just squeeze the biggest one)
            
            reduction_ratio = 0.7 # Aggressive reduction
            current_len = len(target["content"])
            target_len = int(current_len * reduction_ratio)
            
            print(f"  [V11] Compressing Chunk {target['id']} ({current_len} -> {target_len} chars)...")
            
            # Retrieve Resident Instance
            slot_id = target.get("slot_id") # We need to pass slot_id out of _execute_single_task
            if not slot_id or not self.slot_manager.slots.get(slot_id):
                print(f"  [V11] Warn: Slot {slot_id} not resident. Skipping compression.")
                break # Escape infinite loop if we can't compress
                
            instance = self.slot_manager.slots[slot_id]
            compress_prompt = [
                {"role": "system", "content": "You are a Technical Editor."},
                {"role": "user", "content": f"The following output is too long for the system context. \n\nTASK: Rewrite it to be approximately {target_len} characters or less, preserving key technical facts.\n\nCONTENT:\n{target['content']}"}
            ]
            
            # Regenerate
            new_content = self._run_inference(instance, compress_prompt, max_tokens=1500)
            
            if len(new_content) < current_len:
                target["content"] = new_content
                print(f"  [V11] Optimization success: {len(target['content'])} chars.")
            else:
                print(f"  [V11] Optimization failed (expert didn't shrink). Force truncating.")
                target["content"] = target["content"][:target_len] # Fallback
                
        return results

    def _synthesize(self, query: str, results: List[Dict]):
        # Tactical Unload: Free Slot 3 to provide 7GB headroom for Synthesis Context/Activations
        # Data confirms 11GB spike, so this is MANDATORY for 32GB systems.
        print("  [V11] Tactically unloading Slot 3 to guarantee Synthesis VRAM headroom...")
        try:
            self.slot_manager.unload_expert(3)
        except: pass

        # Force cleanup of previous specialized contexts to free VRAM for synthesis
        import gc
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Truncate inputs to prevent Context Overflow / Garbage Generation on DML
        # NOTE: Arbitrary truncation removed in favor of Auto-Compression Loop in _execute_single_task
        combined_data = "\n".join([f"--- [Expert: {r['expert']}] ---\n{r['content']}" for r in results])
        
        # Data Forensics: Dump raw inputs to see true scale
        print(f"\n[V11 Forensics] Combined Data Size: {len(combined_data)} characters")
        print(f"[V11 Forensics] Estimated Tokens: ~{len(combined_data) // 4}")
        
        with open("expert_raw_dump.txt", "w", encoding="utf-8") as f:
            f.write(combined_data)
            
        messages = [
            {"role": "system", "content": self.SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUERY: {query}\n\nDATA:\n{combined_data}"}
        ]
        return self._run_inference(self.supervisor_model, messages, max_tokens=3072)

    def _query_supervisor(self, prompt):
        return self._run_inference(self.supervisor_model, [{"role": "user", "content": prompt}], max_tokens=2048)

    def _run_inference(self, instance, messages, max_tokens=1024):
        return instance.model.generate(messages=messages, temperature=instance.temperature, max_tokens=max_tokens)

    def _parse_chunks(self, text):
        data = self._parse_json(text)
        chunks = []
        if data and "chunks" in data:
            for i, c in enumerate(data["chunks"]):
                if isinstance(c, str):
                    chunks.append(TaskChunk(id=f"c{i+1}", description=c, assigned_expert="python_expert", required_capabilities=[]))
                elif isinstance(c, dict):
                    desc = c.get("description") or c.get("task") or str(c)
                    expert = c.get("assigned_expert") or "python_expert"
                    
                    # FORCE STRING CONVERSION (Prevents unhashable list error)
                    if isinstance(expert, list) and expert: expert = expert[0]
                    expert_id = str(expert)
                    chunk_id = str(c.get("id", f"c{i+1}"))
                    
                    chunks.append(TaskChunk(
                        id=chunk_id, 
                        description=str(desc), 
                        assigned_expert=expert_id, 
                        required_capabilities=[]
                    ))
        
        if not chunks:
            # Plan B: Treat whole response as one chunk
            chunks.append(TaskChunk("c1", str(text), "python_expert", []))
            
        return chunks

    def _parse_json(self, text):
        try:
            if "```json" in text: text = text.split("```json")[1].split("```")[0]
            elif "```" in text: text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except: return None
