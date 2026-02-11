"""
Supervisor Router V9 - Single Executive Intelligence
------------------------------------------------------

⚠️  DEPRECATION WARNING ⚠️
This module is DEPRECATED as of 2026-02-01.
For new development, use:
- `MemoryVectorRouter` from `src.memory_router` for memory-aware routing
- `VectorRouter` from `src.vector_router` for FAISS-based similarity search

Kept for backward compatibility with:
- src/harness_v9.py
- src/harness_v10.py
- src/harness_v11.py

Architecture:
- Single NPU Supervisor (gpt-oss-sg:20b)
- Dedicated methods for Decomposition, Checkpoint, and Synthesis
- Zero legacy Council code
"""

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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

    def __post_init__(self):
        # Mandatory sanitization for College of Experts V11 stability
        self.id = str(self.id)
        
        # Catch empty lists or malformed LLM outputs
        exp = self.assigned_expert
        if not exp or exp == [] or exp == "[]":
            self.assigned_expert = "python_expert"
        elif isinstance(exp, list):
            self.assigned_expert = str(exp[0]) if exp else "python_expert"
        else:
            self.assigned_expert = str(exp)
            
        # Final safety check against the string "[]" leak
        if self.assigned_expert == "[]":
            self.assigned_expert = "python_expert"

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

class SupervisorRouter:
    """
    Executive Supervisor running on NPU (gpt-oss-sg:20b).
    Responsible for high-level planning, quality control, and synthesis.
    """
    
    # --- PROMPTS ---
    
    DECOMPOSE_PROMPT = """You are the Lead Technical Architect. Break down the user's request into distinct, actionable implementation tasks.

    AVAILABLE EXPERTS (Capabilities):
    1. 'python' (Python Specialist): Software engineering, FastAPI/Flask, debugging, testing, API glue.
    2. 'sql' (Database Expert): PostgreSQL/MySQL, schema design, migrations, query optimization, normalization.
    3. 'security' (Security Analyst): OWASP, authentication (OAuth/JWT), encryption, audit logs, vulnerability checks.
    4. 'legal' (Legal Analyst): Regulatory compliance (HIPAA/GDPR), contracts, disclaimers, jurisdiction checks.
    5. 'medical' (Medical Specialist): Clinical diagnosis, pharmacology, anatomy, treatment protocols.
    6. 'math' (Math Specialist): Calculus, statistics, algebra, proofs, complex logic.
    7. 'architecture' (System Architect): Microservices, design patterns, scalability, system boundaries.
    8. 'reasoning' (General Reasoner): Synthesis, logic, cross-domain analysis.

    CRITICAL RULES:
    1. BE CONCISE. The NPU is slow. Minimal tokens.
    2. MAXIMIZE DIVERSITY: Do not assign everything to 'python'. Isolate specific concerns.
    3. DATABASE FIRST: If data storage is needed, create a dedicated 'sql' chunk.
    4. SECURITY PRIORITY: If 'audit' or 'secure' is mentioned, create a dedicated 'security' chunk.
    5. Compliance matches 'legal'.
    6. 'medical' is ONLY for clinical questions, NOT for building medical software.

    For each chunk, provide:
    1. ID (chunk_1 ...)
    2. Description (Brief instruction)
    3. Capabilities (Select strictly from the list above)
    4. Specialist Persona (e.g. "Senior Database Architect")
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
    
    CRITICAL: BE CONCISE. OUTPUT JSON ONLY.

    Respond in JSON:
    ```json
    {{
      "approved": true/false,
      "quality_score": 0.0-1.0,
      "feedback": "..." // Max 1 sentence
    }}
    ```"""

    SYNTHESIS_PROMPT = """You are the Delivery Manager. Combine the expert contributions into a final, coherent response.

    Original Query: {query}
    
    Expert Contributions:
    {contributions}

    Task:
    1. Synthesize all parts into a unified solution.
    2. Ensure smooth transitions between sections.
    3. Verify the final output answers the user's request completely.
    4. BE CONCISE. Avoid fluff.

    Response:"""

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

    def decompose(self, task: str, on_expert_identified=None) -> List[TaskChunk]:
        print(f"[Supervisor] Decomposing task: {task[:50]}...")
        prompt = self.DECOMPOSE_PROMPT.format(query=task)
        
        response = self.generate(prompt, max_tokens=2048)
        
        # Lazy load trigger
        if on_expert_identified: on_expert_identified(response)
        
        data = self._parse_json(response)
        chunks = []
        
        if data and "chunks" in data:
            for c in data["chunks"]:
                # Sanity check capabilities
                caps = c.get("capabilities", [])
                
                # Hard override for medical safety
                desc_lower = c.get("description", "").lower()
                if "medical" in caps and not any(k in desc_lower for k in ["diagnose", "treatment", "clinical", "patient care"]):
                    caps = [x for x in caps if x != "medical"]
                    if "schema" in desc_lower or "database" in desc_lower: caps.append("sql")
                    if "compliance" in desc_lower: caps.append("legal")
                
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
        else:
            print("[Supervisor] Failed to parse chunks. Creating fallback python chunk.")
            # Fallback
            chunk = TaskChunk(
                id=self._next_id(),
                description=task,
                required_capabilities=["python"],
                specialist_persona="Expert Python Developer",
                priority=1
            )
            chunks.append(chunk)
            
        return chunks

    def checkpoint(self, chunk: TaskChunk, work_content: str) -> CheckpointResult:
        print(f"[Supervisor] Reviewing chunk {chunk.id}...")
        prompt = self.CHECKPOINT_PROMPT.format(
            chunk_description=chunk.description,
            work_content=work_content
        )
        response = self.generate(prompt, max_tokens=512)
        data = self._parse_json(response)
        
        if data:
            return CheckpointResult(
                approved=data.get("approved", False),
                feedback=data.get("feedback"),
                quality_score=data.get("quality_score", 0.0)
            )
        return CheckpointResult(False, "Parse Error in Checkpoint", 0.0)

    def synthesize(self, query: str, contributions: str) -> str:
        """Synthesize final response from expert contributions."""
        print(f"[Supervisor] Synthesizing final response...")
        prompt = self.SYNTHESIS_PROMPT.format(query=query, contributions=contributions)
        return self.generate(prompt, max_tokens=2048)

    def assign_chunks(self, chunks: List[TaskChunk]):
        # Simple mapping logic
        keyword_map = {
            "security_expert": ["security", "audit", "auth"],
            "legal_expert": ["legal", "compliance", "hipaa", "gdpr", "disclaimer", "regulation"],
            "python_expert": ["python", "code", "backend", "api", "script"],
            "sql_expert": ["sql", "database", "schema", "query"],
            "medical_expert": ["clinical", "treatment", "diagnosis", "disease"] 
        }
        
        for chunk in chunks:
            assigned = None
            
            # 1. Exact Cap match
            for cap in chunk.required_capabilities:
                expert_id = f"{cap}_expert"
                # Check if this expert type exists in our map
                if any(k.startswith(cap) for k in keyword_map.keys()):
                     assigned = f"{cap}_expert"
                     break
            
            # 2. Keyword match (Fallback)
            if not assigned:
                desc = chunk.description.lower()
                for expert, keys in keyword_map.items():
                    if any(k in desc for k in keys):
                        assigned = expert
                        break
            
            # 3. Default
            chunk.assigned_expert = assigned or "python_expert" 

    def _parse_json(self, text: str) -> Optional[Dict]:
        if not text: return None
        try:
            # Simple cleaning
            cleaned = text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            return json.loads(cleaned.strip())
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
