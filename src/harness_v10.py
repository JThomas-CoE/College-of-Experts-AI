
import time
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from dataclasses import dataclass, field

from .crosstalk_v8 import CrosstalkBus
from .expert_slots_v8 import ExpertSlotManager
from .router_v9 import TaskChunk # Reusing data structure

class HarnessV10:
    """
    V10 'Deep-Reasoning Supervisor' Harness.
    - Router: DeepSeek-R1-Distill-Qwen-7B (Slot 0, GPU)
    - Architecture: 4-Phase GPU Workflow
    """
    
    DECOMPOSE_PROMPT = """Role: Chief Technical Architect
    Task: Decompose the user's request into actionable chunks for specialized experts.
    
    PRINCIPLES OF ARCHITECTURE:
    1. **Capability Matches Action**: 
       - If the user asks to "Write a Script/Code", assign `python_expert`.
       - If the user asks for "Legal Text/Disclaimer", assign `legal_expert`.
    2. **Deep Domain Logic**:
       - **HIPAA/GDPR/Compliance** is a **LEGAL** task -> `legal_expert`.
       - **Clinical/Diagnosis** is a **MEDICAL** task -> `medical_expert`.
    3. **Collaborative Workflow**: 
       - Example: "HIPAA Audit Script" -> 
         1. `legal_expert` defines the compliance checklist.
         2. `python_expert` writes the code to implement that checklist.
    
    Experts Available:
    - python_expert (Software Engineering, APIs, Scripting)
    - security_expert (Auth, Cryptography, Security Audits)
    - sql_expert (Database Schema, Query Optimization)
    - legal_expert (Laws, HIPAA, GDPR, Compliance, Disclaimers)
    - medical_expert (Diagnosis, Anatomy, Pharmacology ONLY)
    
    Output JSON:
    {{
      "chunks": [
        {{ "id": "chunk_1", "description": "...", "assigned_expert": "expert_id", "dependencies": [] }}
      ]
    }}
    
    Query: {query}
    """
    
    CRITIQUE_PLAN_PROMPT = """Role: Project Manager
    Task: Review this project plan. 
    1. Are the experts assigned correctly? (e.g. SQL for DBs, Security for Auth)
    2. Are there missing steps?
    
    Plan: {plan_json}
    
    Output JSON (Approved=True or Corrected Plan):
    {{
       "approved": boolean,
       "feedback": "...",
       "corrected_plan": null // or new JSON like above if major changes
    }}
    """

    def __init__(self, config: Any):
        self.config = config
        self.catalog = config.catalog
        
        # V10: Router is on GPU Slot 0 (DeepSeek)
        # We assume the user has added 'deepseek_reasoner' to catalog pointing to the DML model
        self.slot_manager = ExpertSlotManager(
            catalog=self.catalog,
            num_slots=8, 
            use_npu_for_router=False, 
            npu_slots=[]
        )
        # Pin mechanism (Proto-implementation): We just manually load it once and keep the reference alive.
        # The SlotManager generally uses LRU, but if we don't touch Slot 0 with other models, it should stay.
        # We will ensure strict ID usage.
        self.bus = CrosstalkBus()
        self.supervisor_model = None

    def initialize(self):
        print("[Harness V10] Initializing DeepSeek Supervisor (Slot 0)...")
        # Ensure deepseek_reasoner is in catalog, else fallback to python_expert for testing
        try:
            instance = self.slot_manager.get_or_load_expert("deepseek_reasoner")
        except:
            print("Warning: 'deepseek_reasoner' not in catalog. Using 'general_reasoner'.")
            instance = self.slot_manager.get_or_load_expert("general_reasoner")
            
        self.supervisor_model = instance
        print(f"[Harness V10] Supervisor Ready: {instance.expert_def.id}")

    def process(self, user_input: str) -> str:
        self.initialize()
        
        # --- Phase 1: Context (Skipped for now, assuming query is clear) ---
        
        # --- Phase 2: Decomposition ---
        print("\n[V10 Phase 2] Decomposing Task (DeepSeek)...")
        chunks = self._decompose(user_input)
        
        # --- Phase 3: Plan Review ---
        print("\n[V10 Phase 3] Critiquing Plan (Self-Correction)...")
        chunks = self._critique_plan(chunks)
        
        # Print Plan
        print(f"\n{'='*20} V10 APPROVED PLAN {'='*20}")
        for c in chunks:
            print(f"[{c.id}] {c.assigned_expert}\n    {c.description[:80]}...")
        print(f"{'='*60}\n")
        
        # --- Phase 4: Execution (Parallel GPU Experts) ---
        print("\n[V10 Phase 4] Executing Plan...")
        results = self._execute_parallel(chunks)
        
        # --- Synthesis ---
        print("\n[V10 Phase 5] Synthesizing Final Output...")
        return self._synthesize(user_input, results)

    def _decompose(self, query: str) -> List[TaskChunk]:
        prompt = self.DECOMPOSE_PROMPT.format(query=query)
        response = self._query_supervisor(prompt)
        return self._parse_chunks(response)
        
    def _critique_plan(self, chunks: List[TaskChunk]) -> List[TaskChunk]:
        # Serialize chunks
        plan_json = json.dumps([{"id": c.id, "desc": c.description, "expert": c.assigned_expert} for c in chunks])
        
        prompt = self.CRITIQUE_PLAN_PROMPT.format(plan_json=plan_json)
        response = self._query_supervisor(prompt)
        
        data = self._parse_json(response)
        if data and data.get("approved") is False and data.get("corrected_plan"):
             print(f"  -> Plan Rejected. Applying correction: {data.get('feedback')}")
             return self._parse_chunks(json.dumps(data["corrected_plan"]))
        
        print("  -> Plan Approved.")
        return chunks

    def _execute_parallel(self, chunks: List[TaskChunk]):
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {executor.submit(self._execute_chunk, c): c for c in chunks}
            for future in as_completed(future_to_chunk):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Chunk Failed: {e}")
                    traceback.print_exc()
        return results

    def _execute_chunk(self, chunk: TaskChunk):
        # V9.1 Logic: Expert Gen -> Reviewer Check -> Finalize
        print(f"  > [Chunk {chunk.id}] Started ({chunk.assigned_expert})")
        
        # For simplicity in V10 prototype, we do 1 pass with high quality
        # (Assuming DeepSeek or powerful Savants)
        instance = self.slot_manager.get_or_load_expert(chunk.assigned_expert)
        
        # Check A2A
        context = ""
        msgs = self.bus.receive(chunk.assigned_expert)
        if msgs: context = "Context:\n" + "\n".join([m.content for m in msgs])
        
        prompt = f"""Task: {chunk.description}
        {context}
        Provide professional, production-ready output."""
        
        messages = [
             {"role": "system", "content": instance.expert_def.system_prompt},
             {"role": "user", "content": prompt}
        ]
        
        # Inference
        content = self._run_inference(instance, messages)
        
        # Broadcast
        self.bus.broadcast(chunk.assigned_expert, f"Done {chunk.id}", {"summary": content[:50]})
        print(f"  ✅ [Chunk {chunk.id}] Complete")
        
        return {"id": chunk.id, "expert": chunk.assigned_expert, "content": content}

    def _synthesize(self, query, results):
        try:
            combined = "\n".join([f"--- {r['expert']} ---\n{r['content']}" for r in results])
            prompt = f"Role: Delivery Manager. Synthesize this into a final answer for: '{query}'\n\nData:\n{combined}"
            return self._query_supervisor(prompt)
        except Exception as e:
            traceback.print_exc()
            return f"Error during synthesis: {e}"

    def _query_supervisor(self, prompt):
        # Use simple unformatted generation or chat template for DeepSeek
        messages = [{"role": "user", "content": prompt}]
        return self._run_inference(self.supervisor_model, messages, max_tokens=2048)

    def _run_inference(self, instance, messages, max_tokens=1024):
        # OGA Backend
        return instance.model.generate(
            messages=messages,
            temperature=instance.temperature,
            max_tokens=max_tokens
        )

    def _parse_chunks(self, text):
        data = self._parse_json(text)
        chunks = []
        if data and "chunks" in data:
            for i, c in enumerate(data["chunks"]):
                chunks.append(TaskChunk(
                    id=c.get("id", f"chunk_{i}"), 
                    description=c.get("description"), 
                    assigned_expert=c.get("assigned_expert"),
                    required_capabilities=[]
                ))
        # Fallback
        if not chunks:
            chunks.append(TaskChunk("c1", text, [], "python_expert"))
        return chunks

    def _parse_json(self, text):
        try:
            if "```json" in text: text = text.split("```json")[1].split("```")[0]
            elif "```" in text: text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except: return None
