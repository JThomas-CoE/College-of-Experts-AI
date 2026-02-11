"""
Agentic Harness - Orchestrates multi-expert workflows.

V7.1 - Dual-Loop Council Supervision with A2A Crosstalk
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from .router_v8 import CouncilRouter, TaskChunk
from .expert_catalog import ExpertCatalog, load_catalog
from .expert_slots_v8 import ExpertSlotManager, ExpertInstance
from .crosstalk_v8 import CrosstalkBus, get_crosstalk_bus, MessageType, PartType, MessagePart
from .council_v8 import CouncilMode, CouncilResult
from .episodic_memory import EpisodicMemory, MemoryItem, MemoryStatus

class SessionState(Enum):
    """States in the harness state machine."""
    IDLE = "idle"
    GATHERING = "gathering"  # Wait for more info
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    REFINING = "refining"   # Cross-critique / Council
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"

@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "expert", "system"
    content: str
    expert_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class Session:
    """Represents an active conversation session."""
    session_id: str
    state: SessionState = SessionState.IDLE
    messages: List[Message] = field(default_factory=list)
    active_experts: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class HarnessConfig:
    """Configuration for the agentic harness."""
    catalog_path: str = "config/expert_catalog.json"
    num_slots: int = 4
    use_npu_for_router: bool = True
    npu_slots: List[int] = field(default_factory=lambda: [0, 1, 2])
    memory_path: str = "data/episodic_memory"

class Harness:
    """
    V8 Executive Harness - Orchestrates the Council of Savants.
    """
    
    def __init__(self, config: Optional[HarnessConfig] = None):
        self.config = config or HarnessConfig()
        
        # 1. Foundation
        self.catalog = load_catalog(self.config.catalog_path)
        self.slot_manager = ExpertSlotManager(
            catalog=self.catalog,
            num_slots=self.config.num_slots,
            use_npu_for_router=self.config.use_npu_for_router,
            npu_slots=self.config.npu_slots
        )
        
        # 2. Logic & Comm
        self.memory = EpisodicMemory(storage_path=self.config.memory_path)
        self.bus = get_crosstalk_bus()
        self.router = CouncilRouter(
            slot_manager=self.slot_manager,
            memory=self.memory,
            catalog=self.catalog
        )
        self.council_mode = CouncilMode(self.slot_manager, self.catalog)
        
        # 3. Grounding
        from .vector_backbone import VectorBackbone
        self.vector_store = VectorBackbone("data/vector_db")
        
        # Active sessions
        self.sessions: Dict[str, Session] = {}
        self._session_counter = 0

    def create_session(self) -> Session:
        self._session_counter += 1
        sid = f"s{self._session_counter}_{int(time.time())}"
        session = Session(session_id=sid)
        self.sessions[sid] = session
        return session

    def process(self, session_id: str, user_input: str) -> str:
        """
        V8 Executive Workflow: Reception -> Council Decompose -> Grounded Savant Exec -> Council Synthesis
        """
        session = self.sessions.get(session_id)
        if not session:
            session = self.create_session()
            session_id = session.session_id

        session.messages.append(Message(role="user", content=user_input))
        
        # 1. READINESS
        history = [f"{m.role}: {m.content}" for m in session.messages]
        readiness = self.router.evaluate_readiness(history)
        
        if readiness["status"] == "GATHERING":
            session.state = SessionState.GATHERING
            response = readiness["response"]
            session.messages.append(Message(role="assistant", content=response))
            return response

        # 2. DECOMPOSITION (with Lazy Loading)
        print(f"[Harness] Decomposing task: {user_input[:50]}...")
        session.state = SessionState.DECOMPOSING
        
        # V8 Lazy Loading Logic: Start loading experts in background while NPU works
        import threading
        preloaded_experts = set()
        
        def lazy_preload(router_response: str):
            # Parse response for identified experts and start loading them on GPU
            # Heuristic: Find words ending in _expert (standard catalog IDs)
            import re
            experts = re.findall(r'(\w+_expert)', router_response)
            for expert_id in experts:
                if expert_id not in preloaded_experts:
                    preloaded_experts.add(expert_id)
                    # Skip the router itself (already loaded)
                    if expert_id == "general_reasoner": continue
                    
                    print(f"  [LazyLoad] Found {expert_id}, pre-loading on GPU...")
                    # Run lazy load in a detached thread to not block the router callback
                    threading.Thread(
                        target=self.slot_manager.get_or_load_expert,
                        args=(expert_id,),
                        daemon=True
                    ).start()

        chunks = self.router.decompose(user_input, on_expert_identified=lazy_preload)
        self.router.assign_chunks(chunks)
        
        # Print Assignments clearly
        print("\n[Harness] Decomposition Plan:")
        for c in chunks:
            print(f"  - [{c.id}] -> {c.assigned_expert}: {c.description[:60]}...")
        print("-" * 60)
        
        # 3. EXECUTION (Parallel with Review/Retry/Escalate)
        print(f"[Harness] Running {len(chunks)} chunks (Parallel Exec-Review-Loop)...")
        session.state = SessionState.EXECUTING
        chunk_results = []
        
        # Parallel execution helper
        def process_chunk_wrapper(chunk):
            import random
            import time
            import traceback
            
            # Jitter to prevent thundering herd on NPU/GPU resources
            time.sleep(random.uniform(0.5, 3.0))
            
            try:
                if not chunk.assigned_expert: return None
                return self._process_chunk_with_lifecycle(session, chunk)
            except Exception as e:
                print(f"  [Chunk {chunk.id}] 💥 THREAD CRASH: {e}")
                traceback.print_exc()
                return None

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Use up to 5 parallel workers as requested
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {executor.submit(process_chunk_wrapper, chunk): chunk for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                result_package = future.result()
                if result_package:
                    chunk_results.append(result_package)

        # 4. SYNTHESIS
        print("[Harness] Final Synthesis...")
        session.state = SessionState.SYNTHESIZING
        
        # Combine all committed work summaries
        contributions = "\n\n".join([
            f"### {r['expert']} (Part {r['chunk_id']}):\n{r['content']}" 
            for r in chunk_results
        ])
        
        final_prompt = f"Original Query: {user_input}\n\nExpert Contributions:\n{contributions}\n\nTask: Provide a unified and thorough final response."
        
        # Supervisor Synthesis (V9)
        final_response = self.router.synthesize(user_input, contributions)
        
        session.state = SessionState.IDLE
        session.messages.append(Message(role="assistant", content=final_response))
        return final_response

    def _process_chunk_with_lifecycle(self, session: Session, chunk: TaskChunk) -> Dict:
        """
        Execute a chunk with the full V8 lifecycle: 
        Attempt 1 (Fast) -> Review -> Attempt 2 (Fix) -> Review -> Escalate to Council (Robust)
        """
        print(f"  [Chunk {chunk.id}] Started -> {chunk.assigned_expert}")
        
        # --- Attempt 1: Fast Single Instance ---
        content = self._query_single_savant_v8(session, chunk)
        
        # Create temp WIP item for review
        wip = MemoryItem(id="temp", expert_id=chunk.assigned_expert, content=content, chunk_ref=chunk.id)
        
        # Review
        review = self.router.checkpoint(chunk, wip)
        
        if review.approved:
            print(f"  [Chunk {chunk.id}] ✅ Approved on Attempt 1")
        else:
            print(f"  [Chunk {chunk.id}] ❌ Rejected (Att 1): {review.feedback}")
            
            # --- Attempt 2: Retry with Feedback ---
            # Append feedback to prompt
            chunk.description += f"\n\n[REVIEW FEEDBACK]: {review.feedback}\nPlease address these issues."
            content = self._query_single_savant_v8(session, chunk)
            wip.content = content
            
            review = self.router.checkpoint(chunk, wip)
            
            if review.approved:
                 print(f"  [Chunk {chunk.id}] ✅ Approved on Attempt 2")
            else:
                 print(f"  [Chunk {chunk.id}] ❌ Rejected (Att 2): {review.feedback}")
                 
                 # --- Attempt 3: Final Retry (Single Expert) ---
                 # No Council Escalation (V9 Simplification)
                 chunk.description += f"\n\n[FINAL FEEDBACK]: {review.feedback}\nEnsure strict adherence."
                 content = self._query_single_savant_v8(session, chunk)
                 
                 # Accept best effort
                 print(f"  [Chunk {chunk.id}] ⚠️ Accepting Result (Attempt 3 Best Effort)")

        # Finalize
        result = content
        
        # A2A Broadcast
        self.bus.broadcast(
            from_expert=chunk.assigned_expert,
            content=f"Completed {chunk.id}",
            data={"chunk_id": chunk.id, "summary": result[:300]}
        )
        
        # Commit to Episodic Memory
        self.memory.add_wip(
            expert_id=chunk.assigned_expert,
            content=result,
            chunk_ref=chunk.id
        )
        
        return {
            "chunk_id": chunk.id,
            "expert": chunk.assigned_expert,
            "content": result
        }

    def _query_single_savant_v8(self, session: Session, chunk: TaskChunk) -> str:
        """Grounded Savant query with A2A Crosstalk (V8)."""
        expert_id = chunk.assigned_expert
        
        # 1. RAG
        domain = expert_id.split("_")[0]
        grounding = self._get_grounded_context(chunk.description, domain)
        
        # 2. A2A
        pending = self.bus.receive(expert_id)
        bus_context = ""
        if pending:
            bus_context = "\nCONTEXT FROM OTHER EXPERTS:\n" + \
                         "\n".join([f"- {m.from_expert}: {m.content}" for m in pending])

        # 3. Prompt
        prompt = chunk.description
        if grounding:
            prompt = f"KNOWLEDGE BASE:\n{grounding}\n\n{prompt}"
        if bus_context:
            prompt = f"{bus_context}\n\n{prompt}"
            
        instance = self.slot_manager.get_or_load_expert(expert_id)
        
        # 4. Two-Tier Persona Injection
        system_prompt = instance.expert_def.system_prompt
        if chunk.specialist_persona:
            system_prompt = f"{system_prompt}\n\nSPECIALIST ROLE: {chunk.specialist_persona}"
            
        # Format for OGA/Backend
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return instance.model.generate(model_id=expert_id, messages=messages, max_tokens=1024)

    def _query_savant_council_v8(self, session: Session, chunk: TaskChunk) -> str:
        """3-member diverse Council of Savants (V8 Booster)."""
        instance = self.slot_manager.get_or_load_expert(chunk.assigned_expert)
        system_prompt = instance.expert_def.system_prompt
        if chunk.specialist_persona:
            system_prompt = f"{system_prompt}\n\nSPECIALIST ROLE: {chunk.specialist_persona}"

        return self.council_mode.run(
            query=chunk.description,
            expert_id=chunk.assigned_expert,
            num_members=3,
            selection_method="vote",
            system_prompt=system_prompt
        ).best_response

    def _get_grounded_context(self, query: str, domain: str) -> str:
        try:
            results = self.vector_store.query(domain, query, n_results=2)
            docs = results.get("documents", [[]])[0]
            if docs:
                return "\n".join([f"- {d}" for d in docs])
        except Exception: pass
        return ""

    def get_status(self) -> dict:
        return {
            "sessions": len(self.sessions),
            "vram": self.slot_manager.get_vram_usage(),
            "bus": self.bus.get_stats(),
            "memory": self.memory.get_stats()
        }

    def cleanup(self):
        self.slot_manager.cleanup()
