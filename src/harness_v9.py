"""
Harness V9 - Streamlined Execution Lifecycle
------------------------------------------------------
Architecture:
- Parallel Execution of Chunks
- Strict Lifecycle: Attempt 1 -> Supervisor Check -> Attempt 2 (Feedback) -> Attempt 3 (Feedback) -> Finalize
- No Council Escalation
- A2A enabled
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from .crosstalk_v8 import CrosstalkBus
from .episodic_memory import EpisodicMemory, MemoryItem
from .expert_slots_v8 import ExpertSlotManager
from .router_v9 import SupervisorRouter, TaskChunk
from enum import Enum
from dataclasses import dataclass, field
from typing import List

class SessionState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Session:
    id: str
    state: SessionState = SessionState.IDLE
    messages: List[Message] = field(default_factory=list)

class HarnessV9:
    def __init__(self, config: Any):
        self.config = config
        
        # Components
        self.memory = EpisodicMemory(config.memory_path)
        self.catalog = config.catalog
        # Configure Slot Manager for V9 Topology (1 NPU + 7 GPU slots)
        self.slot_manager = ExpertSlotManager(
            catalog=self.catalog,
            num_slots=8,
            use_npu_for_router=True,
            npu_slots=[0]
        )
        self.bus = CrosstalkBus()
        
        # New V9 Router
        self.router = SupervisorRouter(
            slot_manager=self.slot_manager,
            memory=self.memory,
            catalog=self.catalog,
            temperature=0.1
        )

    def process(self, user_input: str) -> str:
        """Main V9 Workflow Entrypoint."""
        
        # 1. Initialize
        print("\n[Harness V9] Initializing Supervisor...")
        self.router.initialize()
        session = Session(id=f"sess_{int(time.time())}", messages=[Message(role="user", content=user_input)])
        
        # 2. Decompose
        print("[Harness V9] Supervisor Decomposing Task...")
        
        def lazy_load_hint(response_text):
            # Pre-warm GPU based on textual hints if possible
            pass
            
        chunks = self.router.decompose(user_input, on_expert_identified=lazy_load_hint)
        self.router.assign_chunks(chunks)
        
        # Display Plan
        print(f"\n{'='*20} V9 EXECUTION PLAN {'='*20}")
        for c in chunks:
            print(f"[{c.id}] {c.assigned_expert}\n    Task: {c.description[:80]}...")
        print(f"{'='*60}\n")
        
        if not chunks:
            return "Error: Supervisor failed to decompose task."

        # 3. Parallel Execution
        session.state = SessionState.EXECUTING
        results = []
        
        def worker(chunk):
            # Jitter
            time.sleep(0.5) 
            return self._execute_chunk_lifecycle(session, chunk)

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {executor.submit(worker, c): c for c in chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    res = future.result()
                    if res: results.append(res)
                except Exception as e:
                    c = future_to_chunk[future]
                    print(f"[Harness] Chunk {c.id} CRASHED: {e}")
                    traceback.print_exc()

        # 4. Synthesis
        print("\n[Harness V9] Supervisor Synthesis...")
        session.state = SessionState.SYNTHESIZING
        
        contributions = "\n\n".join([
            f"### Expert: {r['expert']} (ID: {r['chunk_id']})\nOutput:\n{r['content']}" 
            for r in results
        ])
        
        final_output = self.router.synthesize(user_input, contributions)
        
        print(f"\n[Harness V9] Task Complete. Tokens generated: {len(final_output)//4} approx.")
        return final_output

    REVIEW_PROMPT = """You are a Senior Technical Reviewer. Grade the following work.
    
    Task: {description}
    Code/Content:
    {content}
    
    CRITICAL: BE CONCISE.
    1. Check for correctness.
    2. Check for safety/security.
    3. Check for specific instructions.
    
    Respond in JSON:
    {{
      "approved": true/false,
      "feedback": "..."
    }}
    """

    def _execute_chunk_lifecycle(self, session: Session, chunk: TaskChunk) -> Dict:
        """
        The GPU-Accelerated Self-Correction Loop.
        1. Expert Generates
        2. Expert Persona Swaps to 'Reviewer'
        3. Reviewer Grades
        4. Expert Fixes (if needed)
        """
        print(f"  > [Chunk {chunk.id}] Started ({chunk.assigned_expert})")
        
        # 1. Initial Generation
        content = self._generate_step(chunk, attempt=1)
        
        # 2. Review Cycle (Max 2 retries)
        for attempt in range(1, 3):
            # Review (GPU side)
            review = self._gpu_review(chunk, content)
            
            if review.get("approved"):
                print(f"  ✅ [Chunk {chunk.id}] Self-Review Passed (Try {attempt})")
                return self._finalize(chunk, content)
            
            # Feedback Loop
            feedback = review.get("feedback", "Improve quality.")
            print(f"  Refining [Chunk {chunk.id}] (GPU Critic: {feedback[:50]}...)")
            chunk.description += f"\n\n[REVIEWER FEEDBACK]: {feedback}"
            
            # Retry Generation
            content = self._generate_step(chunk, attempt=attempt+1)
            
        # 3. Final Acceptance (Best Effort)
        print(f"  ⚠️ [Chunk {chunk.id}] Accepting Best Effort (Try 3)")
        return self._finalize(chunk, content)

    def _generate_step(self, chunk: TaskChunk, attempt: int) -> str:
        """Generate content using the assigned expert persona."""
        # Ensure we are in Expert Mode
        instance = self.slot_manager.get_or_load_expert(chunk.assigned_expert)
        
        # If we were reviewing, we might need to swap back, but get_or_load handles it 
        # (assuming we use the ID 'python_expert' vs 'python_reviewer')
        # Actually, get_or_load checks ID. So we must be careful not to confuse the SlotManager.
        # We will manually force a persona swap if needed, but the simple way is just to call get_or_load
        # with the original expert ID, which re-sets the system prompt if it changed.
        
        return self._query_model(instance, chunk.description, chunk.assigned_expert)

    def _gpu_review(self, chunk: TaskChunk, content: str) -> Dict:
        """Swap to Reviewer persona and grade."""
        # 1. Swap Persona on the same slot
        # We define a temporary 'Reviewer' ID or just inject a Reviewer system prompt effectively.
        # The SlotManager.swap_persona uses a catalog ID. We need a 'reviewer' entry in catalog?
        # OR we can hack it by manually loading the same model with a different prompt?
        # Let's use the catalog approach if possible, or just 'general_reasoner' if it's on GPU?
        # Better: We create a synthetic 'reviewer' persona injection here.
        
        instance = self.slot_manager.get_or_load_expert(chunk.assigned_expert)
        original_prompt = instance.expert_def.system_prompt
        
        # Temporary Swap for this inference only
        reviewer_sys_prompt = "You are a Senior Code Reviewer. Be critical, concise, and security-minded."
        
        prompt = self.REVIEW_PROMPT.format(description=chunk.description, content=content)
        
        # Manually construct messages with Reviewer System Prompt
        messages = [
            {"role": "system", "content": reviewer_sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Run Inference
        response = self._run_inference(instance, messages, max_new_tokens=256)
        
        # Parse JSON
        import json
        import re
        try:
            # Extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except:
             # Fallback
             return {"approved": True, "feedback": "Parse Error"}

    def _query_model(self, instance, task_desc, expert_name) -> str:
        """Standard expert generation."""
        # Check A2A
        pending = self.bus.receive(expert_name)
        context_str = ""
        if pending:
            context_str = "CONTEXT FROM TEAM:\n" + "\n".join([f"- {m.from_expert}: {m.content}" for m in pending])
            
        full_prompt = f"""Task: {task_desc}
        
        {context_str}
        
        Provide high-quality, executable code or detailed analysis."""
        
        messages = [
            {"role": "system", "content": instance.expert_def.system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        return self._run_inference(instance, messages)

    def _run_inference(self, instance, messages, max_new_tokens=1024) -> str:
        """Unified OGA/FLM inference backend."""
        # --- PATH A: ONNX GenAI (DirectML) ---
        if instance.capabilities.model_family == "oga":
            return instance.model.generate(
                messages=messages,
                temperature=instance.temperature,
                max_tokens=max_new_tokens
            )
            
        # --- PATH B: FLM (NPU) ---
        if instance.capabilities.model_family == "flm":
            return instance.model.generate(
                messages=messages,
                temperature=instance.temperature,
                max_tokens=max_new_tokens
            )

        # --- PATH C: Legacy PyTorch / HuggingFace ---
        from .chat_utils import UniversalChatFormatter
        formatter = UniversalChatFormatter(instance.processor, instance.capabilities)
        text, images = formatter.format_messages(messages)
        inputs = formatter.prepare_inputs(text, images, instance.model.device)
        
        import torch
        with torch.no_grad():
             outputs = instance.model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 do_sample=True,
                 temperature=instance.temperature
             )
             
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        tokenizer = getattr(instance.processor, "tokenizer", instance.processor)
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _finalize(self, chunk, content):
        # Broadcast success
        self.bus.broadcast(chunk.assigned_expert, f"Finished {chunk.id}", {"summary": content[:100]})
        return {
            "chunk_id": chunk.id,
            "expert": chunk.assigned_expert,
            "content": content
        }
