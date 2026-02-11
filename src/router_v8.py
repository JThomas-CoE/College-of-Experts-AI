"""
Supervisor Router V8 - Council Router Implementation
------------------------------------------------------

⚠️  DEPRECATION WARNING ⚠️
This module is DEPRECATED as of 2026-02-01.
For new development, use:
- `MemoryVectorRouter` from `src.memory_router` for memory-aware routing
- `VectorRouter` from `src.vector_router` for FAISS-based similarity search

Kept for backward compatibility with:
- src/harness.py
- src/__init__.py

Architecture:
- Single NPU Supervisor (Shifted from Voting Council)
- Dedicated methods for Decomposition, Checkpoint, and Synthesis
- Simplified Slot Management interactions
"""

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

from .expert_catalog import ExpertCatalog
from .expert_slots_v8 import ExpertSlotManager
from .episodic_memory import EpisodicMemory

@dataclass
class TaskChunk:
    id: str
    description: str
    required_capabilities: List[str]
    assigned_expert: Optional[str] = None
    specialist_persona: Optional[str] = None
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"

@dataclass
class CompletionStatus:
    is_complete: bool
    quality_score: float
    missing_aspects: List[str]
    reasoning: str

@dataclass
class CheckpointResult:
    approved: bool
    feedback: Optional[str]
    quality_score: float

class CouncilRouter:
    """
    Executive Supervisor running on NPU (gpt-oss-sg:20b).
    Responsible for high-level planning, quality control, and synthesis.
    """
    
    # --- PROMPTS ---
    
    DECOMPOSE_PROMPT = """You are the Lead Technical Architect. Break down the user's request into distinct, actionable implementation tasks.

    CRITICAL RULE: Do NOT assign the 'medical' capability for tasks involving software, data schema, databases, HIPAA compliance, or record management. 
    - Use 'legal' for compliance/disclaimers.
    - Use 'python' or 'sql' for backend/database work.
    - Use 'security' for audits.
    - ONLY use 'medical' if the task requires clinical diagnosis or treatment knowledge.

    For each chunk, provide:
    1. ID (chunk_1 ...)
    2. Description (Detailed instruction)
    3. Capabilities (python, sql, security, legal, etc.)
    4. Specialist Persona (e.g. "Senior Database Architect specializing in HIPAA schemas")
    5. Priority

    Query: {query}
    
    Respond in JSON:
    ```json
    {{
      "chunks": [
        {{ "id": "chunk_1", "description": "...", "capabilities": ["..."], "specialist_persona": "...", "priority": 1 }}
      ]
    }}
    ```"""

    CHECKPOINT_PROMPT = """You are a Quality Assurance Specialist. Review the work submitted by an expert.

    Original Task: {chunk_description}
    Expert Work:
    {work_content}

    Evaluate:
    1. Did they follow instructions?
    2. Is the code/content valid and safe?
    
    Respond in JSON:
    ```json
    {{
      "approved": true/false,
      "quality_score": 0.0-1.0,
      "feedback": "..." // Required if rejected
    }}
    ```"""

    COMPLETION_PROMPT = """Evaluate if the overall project is complete.

    Original Query: {query}
    Committed Work Summary:
    {committed_work}

    Respond in JSON:
    ```json
    {{
      "is_complete": true/false,
      "quality_score": 0.0-1.0,
      "missing_aspects": [],
      "reasoning": "..."
    }}
    ```"""

    def __init__(
        self,
        slot_manager: ExpertSlotManager,
        memory: EpisodicMemory,
        catalog: ExpertCatalog,
        temperature: float = 0.1
    ):
        self.slot_manager = slot_manager
        self.memory = memory
        self.catalog = catalog
        self.temperature = temperature
        
        self.supervisor_model = None
        self._initialized = False
        self._lock = threading.RLock()
        
        # State
        self.chunks: Dict[str, TaskChunk] = {}
        self._chunk_counter = 0

    def initialize(self):
        """Load the Supervisor model on NPU (Slot 0)."""
        if self._initialized: return
        
        with self._lock:
            print("[Supervisor] Initializing NPU Executive...")
            # We request 'general_reasoner'. The SlotManager/Factory handles the mapping to gpt-oss-sg:20b
            # because we fixed the ModelFactory.
            instance = self.slot_manager.get_or_load_expert("general_reasoner", self.temperature)
            self.supervisor_model = instance.model
            self._initialized = True
            print(f"[Supervisor] Ready. Backend: {getattr(self.supervisor_model, 'config', 'Unknown')}")

    def _ensure_ready(self):
        if not self._initialized: self.initialize()

    def generate(self, prompt: str, max_tokens=1024) -> str:
        """Helper for single-turn generation."""
        self._ensure_ready()
        try:
            return self.supervisor_model.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"[Supervisor] Generation Error: {e}")
            return ""

    def decompose(self, task: str, image=None, on_expert_identified=None) -> List[TaskChunk]:
        print(f"[Supervisor] Decomposing task: {task[:50]}...")
        prompt = self.DECOMPOSE_PROMPT.format(query=task)
        
        response = self.generate(prompt, max_tokens=2048)
        
        # Lazy load trigger
        if on_expert_identified: on_expert_identified(response)
        
        data = self._parse_json(response)
        if not data or "chunks" not in data:
            print("[Supervisor] Failed to parse decomposition. Using fallback.")
            return [] # Logic elsewhere handles fallback?
            
        chunks = []
        for c in data["chunks"]:
            # Sanity check capabilities
            caps = c.get("capabilities", [])
            # Hard override for medical if it slipped through
            if "medical" in caps and not any(k in c.get("description", "").lower() for k in ["diagnose", "treatment", "clinical"]):
                caps = [x for x in caps if x != "medical"]
                if "schema" in c.get("description", "").lower(): caps.append("sql")
                if "compliance" in c.get("description", "").lower(): caps.append("legal")
            
            chunk = TaskChunk(
                id=self._next_id(),
                description=c.get("description"),
                required_capabilities=caps,
                specialist_persona=c.get("specialist_persona"),
                priority=c.get("priority", 1),
                dependencies=c.get("dependencies", [])
            )
            self.chunks[chunk.id] = chunk
            chunks.append(chunk)
            
        return chunks

    def checkpoint(self, chunk: TaskChunk, work_item: Any) -> CheckpointResult:
        print(f"[Supervisor] Reviewing chunk {chunk.id}...")
        prompt = self.CHECKPOINT_PROMPT.format(
            chunk_description=chunk.description,
            work_content=work_item.content
        )
        response = self.generate(prompt, max_tokens=512)
        data = self._parse_json(response)
        
        if data:
            return CheckpointResult(
                approved=data.get("approved", False),
                feedback=data.get("feedback"),
                quality_score=data.get("quality_score", 0.0)
            )
        return CheckpointResult(False, "Parse Error", 0.0)

    def completion_eval(self, query: str, committed_work: str) -> CompletionStatus:
        prompt = self.COMPLETION_PROMPT.format(query=query, committed_work=committed_work)
        response = self.generate(prompt, max_tokens=512)
        data = self._parse_json(response)
        
        if data:
            return CompletionStatus(
                is_complete=data.get("is_complete", False),
                quality_score=data.get("quality_score", 0.0),
                missing_aspects=data.get("missing_aspects", []),
                reasoning=data.get("reasoning", "")
            )
        return CompletionStatus(False, 0.0, [], "Parse Error")

    def synthesize(self, query: str, contributions: str) -> str:
        """Synthesize final response from expert contributions."""
        print(f"[Supervisor] Synthesizing final response...")
        prompt = self.SYNTHESIS_PROMPT.format(query=query, contributions=contributions)
        return self.generate(prompt, max_tokens=2048)

    def assign_chunks(self, chunks: List[TaskChunk]):
        # Simple mapping logic
        keyword_map = {
            "security_expert": ["security", "audit", "auth"],
            "legal_expert": ["legal", "compliance", "hipaa", "gdpr", "disclaimer"],
            "python_expert": ["python", "code", "backend", "api"],
            "medical_expert": ["clinical", "treatment", "diagnosis"] # Strict subset
        }
        
        for chunk in chunks:
            assigned = None
            # 1. Cap match
            for cap in chunk.required_capabilities:
                if cap == "legal": assigned = "legal_expert"
                elif cap == "security": assigned = "security_expert"
                elif cap == "python": assigned = "python_expert"
                elif cap == "medical": assigned = "medical_expert" # Only if explicit
            
            # 2. Keyword match (Fallback)
            if not assigned:
                desc = chunk.description.lower()
                for expert, keys in keyword_map.items():
                    if any(k in desc for k in keys):
                        assigned = expert
                        break
            
            chunk.assigned_expert = assigned or "python_expert" # Default

    def _parse_json(self, text: str) -> Optional[Dict]:
        try:
            # Simple cleaning
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except:
            return None

    def _next_id(self):
        self._chunk_counter += 1
        return f"chunk_{self._chunk_counter:03d}"

    def reset(self):
        self.chunks.clear()
        self._chunk_counter = 0

    def cleanup(self):
        with self._lock:
            self.supervisor_model = None
            self._initialized = False
