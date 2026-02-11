# College of Experts V7 - Full Specification
## Codename: "HRM Council"

**Version**: 7.0  
**Date**: 2026-01-17  
**Status**: DRAFT - Awaiting Approval  

---

## 1. Executive Summary

V7 transforms the College of Experts from a parallel one-shot system (V6) into a hierarchical, iterative multi-agent collaboration framework with:

- **Dynamic Expert Loading**: Hot-swap experts from disk library
- **HRM Structure**: Parallel chunks with periodic crosstalk + Router checkpoints
- **Temperature-Diverse Councils**: Multiple copies of same expert at varied temperatures
- **Episodic Memory**: Committed (immutable) vs WIP (mutable) work tracking
- **A2A-Inspired Protocol**: Standardized expert-to-expert communication

### Implementation Constraint (V7.0)

> **Note**: Initial implementation uses **Qwen3-VL:4B** with specialized harness prompts to simulate different experts. True model swapping (loading different specialized models from disk) will be added in a future iteration. The architecture is designed to support both approaches transparently.

---

## 2. Architecture Overview (Dual-Loop HRM)

The system operates with two nested feedback loops:

- **Outer Loop (Task Level)**: Router receives committed work from memory, evaluates task completion, and either synthesizes final output OR re-decomposes for another round.
- **Inner Loop (Chunk Level)**: Each expert iterates on its assigned chunk with crosstalk and checkpoints until the chunk is approved.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                                                          │
                    │  ╔═══════════════════════════════════════════════════╗  │
                    │  ║            OUTER LOOP (Task Level)                ║  │
                    │  ║  Router evaluates: COMPLETE → Synthesize & Exit   ║  │
                    │  ║                     INCOMPLETE → Re-decompose     ║  │
                    │  ╚═══════════════════════════════════════════════════╝  │
                    │                          │                               │
                    ▼                          ▼                               │
┌─────────────────────────────────────────────────────────────────┐           │
│                         ROUTER (Supervisor)                      │           │
│  • Task decomposition        • Expert routing                    │           │
│  • Checkpoint evaluation     • Commit approval                   │           │
│  • Completion detection      • Final synthesis                   │           │
└───────────────────────────────┬─────────────────────────────────┘           │
                                │                                              │
         ┌──────────────────────┼──────────────────────┐                      │
         │                      │                      │                      │
    ┌────▼────┐            ┌────▼────┐           ┌────▼────┐                 │
    │  Slot 0 │◄──────────►│  Slot 1 │◄─────────►│  Slot 2 │                 │
    │ Expert  │  Crosstalk │ Expert  │ Crosstalk │ Expert  │                 │
    │ ┌─────┐ │            │ ┌─────┐ │           │ ┌─────┐ │                 │
    │ │INNER│ │            │ │INNER│ │           │ │INNER│ │                 │
    │ │LOOP │ │            │ │LOOP │ │           │ │LOOP │ │                 │
    │ └──┬──┘ │            │ └──┬──┘ │           │ └──┬──┘ │                 │
    └────┼────┘            └────┼────┘           └────┼────┘                 │
         │                      │                      │                      │
         │  ╔═══════════════════════════════════════╗ │                      │
         │  ║     INNER LOOP (Chunk Level)          ║ │                      │
         │  ║  Work → Crosstalk → Checkpoint →      ║ │                      │
         │  ║  Approved? → Commit : Iterate         ║ │                      │
         │  ╚═══════════════════════════════════════╝ │                      │
         │                      │                      │                      │
         └──────────────────────┴──────────────────────┘                      │
                                │                                              │
                    ┌───────────▼───────────┐                                 │
                    │   EPISODIC MEMORY     │                                 │
                    │ ┌─────────┬─────────┐ │                                 │
                    │ │Committed│   WIP   │ │─────────────────────────────────┘
                    │ │(frozen) │(mutable)│ │  ◄── OUTER LOOP FEEDBACK
                    │ └─────────┴─────────┘ │      (committed work → Router)
                    └───────────────────────┘
```

### Loop Descriptions

**OUTER LOOP (Ralph Loop / Task Completion)**
1. Router decomposes task → assigns chunks to experts
2. Experts execute (via inner loops)
3. Committed results flow to memory
4. Router queries memory: "Is task complete?"
   - **YES**: Synthesize final answer, exit
   - **NO**: Identify gaps, re-decompose, assign new chunks, repeat

**INNER LOOP (Expert Chunk Refinement)**
1. Expert generates initial chunk output
2. Crosstalk with peers (request/provide input)
3. Router checkpoint: review quality
   - **APPROVED**: Commit to memory, expert done with chunk
   - **REJECTED**: Expert iterates, repeat from step 1

---

## 3. Component Specifications

### 3.1 Expert Library

**Location**: `models/experts/`

**Structure**:
```
models/experts/
├── catalog.json              # Expert registry
├── python_expert_2b/         # Specialized Python model
├── security_expert_2b/       # Security-focused model
├── sql_expert_1b/            # SQL specialist
├── creative_writer_4b/       # Creative writing model
├── math_reasoner_4b/         # Math and logic
└── general_reasoner_7b/      # Cross-domain synthesis
```

**Catalog Schema** (`catalog.json`):
```json
{
  "experts": [
    {
      "id": "python_expert",
      "name": "Python Specialist",
      "model_path": "models/experts/python_expert_2b",
      "capabilities": ["python", "fastapi", "django", "debugging"],
      "vram_mb": 2048,
      "recommended_temp": 0.3,
      "tools": ["code_wiki", "pypi_docs"]
    }
  ]
}
```

### 3.2 Expert Slot Manager

**File**: `src/expert_slots.py`

**Responsibilities**:
- Load/unload experts dynamically
- Track VRAM usage
- Manage hot slots (in-memory) vs cold (on-disk)
- LRU eviction when VRAM full

**Interface**:
```python
class ExpertSlotManager:
    def __init__(self, max_vram_mb: int, num_slots: int):
        """Initialize with VRAM budget and slot count."""
    
    def load_expert(self, expert_id: str, temperature: float = None) -> ExpertInstance:
        """Load expert into a slot. Evict LRU if needed."""
    
    def unload_expert(self, slot_id: int) -> None:
        """Free slot for reuse."""
    
    def get_loaded_experts(self) -> List[ExpertInstance]:
        """Return currently loaded experts."""
    
    def get_vram_usage(self) -> Dict[str, int]:
        """Return per-slot VRAM usage."""
```

### 3.3 Crosstalk Protocol (A2A-Aligned)

**File**: `src/crosstalk.py`

**V7.1 Update**: Now uses A2A-style multi-part messages to minimize tokenization overhead.

**Message Parts** (inspired by Google A2A):
```python
class PartType(Enum):
    TEXT = "text"       # Natural language (requires tokenization)
    DATA = "data"       # Structured data (NO tokenization!)
    ARTIFACT = "artifact"  # Memory reference (NO tokenization!)
    CODE = "code"       # Code with language tag
```

**Message Format**:
```python
@dataclass
class CrosstalkMessage:
    from_expert: str
    to_expert: str  # or "broadcast"
    msg_type: Literal["request", "response", "notify", "critique"]
    parts: List[MessagePart]  # A2A-style multi-part content
    
    # Efficiency: Only TEXT parts require tokenization
    # DATA and ARTIFACT parts are accessed directly in Python
```

**Efficiency Comparison**:
| Message Type | Tokenization Cost | Use Case |
|--------------|-------------------|----------|
| Text-only (legacy) | 100% | Backward compatibility |
| Structured (A2A) | ~10-20% | Recommended for efficiency |
| Artifact-only | 0% | Passing memory references |

**Interface**:
```python
class CrosstalkBus:
    def send_structured(
        self, from_expert: str, to_expert: str,
        summary: str,  # Brief text (~10-20 tokens)
        data: dict     # Detailed data (NOT tokenized!)
    ) -> str:
        """Send message with minimal tokenization overhead."""
    
    def receive(self, expert_id: str) -> List[CrosstalkMessage]:
        """Receive pending messages."""
        # msg.get_text_parts()  -> requires tokenization
        # msg.get_data_parts()  -> direct Python access!
        # msg.get_artifact_refs() -> memory lookups
```

### 3.4 Episodic Memory

**File**: `src/episodic_memory.py`

**Partitions**:
- **Committed**: Approved work, immutable, queryable by all experts
- **WIP**: Work in progress, mutable only by owning expert

**Interface**:
```python
class EpisodicMemory:
    def __init__(self, backend: Literal["memvid", "json", "sqlite"]):
        """Initialize with storage backend."""
    
    def add_wip(self, expert_id: str, content: str, metadata: dict) -> str:
        """Add work-in-progress item. Returns item_id."""
    
    def commit(self, item_id: str) -> bool:
        """Move WIP to committed (requires Router approval)."""
    
    def query_committed(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search committed work."""
    
    def get_wip(self, expert_id: str) -> List[MemoryItem]:
        """Get WIP for specific expert."""
    
    def rollback(self, item_id: str) -> bool:
        """Discard WIP item."""
```

### 3.5 Council Router (HRM Supervisor)

**File**: `src/router_v7.py`

The Router uses a **Council of 3× Qwen3-VL:4B** with temperature-diverse parallel inference.

**Why Council Router:**
- **Vision-capable**: Can analyze images/diagrams for task decomposition
- **Parallel inference**: Saturates GPU compute efficiently
- **VRAM efficient**: 3× 4B (~12GB) vs Nemotron 30B (~29GB)
- **Error reduction**: Voting suppresses single-instance mistakes

**Architecture:**
```
┌────────────────────────────────────────────────────────────────┐
│                    COUNCIL ROUTER (3× Qwen3-VL:4B)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   T=0.3     │  │   T=0.5     │  │   T=0.7     │            │
│  │Conservative │  │  Balanced   │  │Exploratory  │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │ PARALLEL       │ INFERENCE      │                    │
│         └────────────────┼────────────────┘                    │
│                          ▼                                     │
│                    MERGE / VOTE                                │
└────────────────────────────────────────────────────────────────┘
```

**Responsibilities**:
- Decompose task into chunks (with vision input support)
- Route chunks to appropriate experts
- Conduct periodic checkpoints (inner loop)
- Approve/reject commits
- Evaluate task completion (outer loop)
- Identify gaps and re-decompose (outer loop)
- Synthesize final output

**Interface**:
```python
class CouncilRouter:
    def __init__(
        self, 
        slot_manager: ExpertSlotManager,
        temperatures: List[float] = [0.3, 0.5, 0.7]
    ):
        """Initialize Council Router with temperature-diverse instances."""
        self.temperatures = temperatures
        self.instances: List[ExpertInstance] = []
    
    def initialize(self) -> None:
        """Load 3× Qwen3-VL instances for parallel inference."""
        for temp in self.temperatures:
            instance = self.slot_manager.load_expert("general_reasoner", temperature=temp)
            self.instances.append(instance)
    
    # --- Task Decomposition (Parallel) ---
    def decompose(self, query: str, image: Optional[Image] = None) -> List[TaskChunk]:
        """
        Decompose query with council voting.
        All 3 instances generate in parallel, results merged.
        """
    
    def assign_chunks(self, chunks: List[TaskChunk]) -> Dict[str, str]:
        """Map chunks to expert_ids. Returns {chunk_id: expert_id}."""
    
    # --- Inner Loop (Chunk Level) ---
    def checkpoint(self, expert_id: str, wip_id: str) -> CheckpointResult:
        """Council evaluates WIP quality, majority vote on approval."""
    
    # --- Outer Loop (Task Level) ---
    def evaluate_completion(self, query: str, committed_work: List[MemoryItem]) -> CompletionStatus:
        """Council votes on task completion (2/3 majority required)."""
    
    def identify_gaps(self, status: CompletionStatus) -> List[TaskChunk]:
        """Identify missing pieces and generate new chunks to fill gaps."""
    
    def is_complete(self) -> bool:
        """Quick check: is the current task satisfactorily complete?"""
    
    # --- Synthesis ---
    def synthesize(self, committed_work: List[MemoryItem]) -> str:
        """Generate final response from committed work (uses T=0.5 instance)."""
    
    # --- Parallel Execution ---
    def _parallel_generate(self, prompt: str, image: Optional[Image] = None) -> List[str]:
        """Run all 3 instances in parallel, return list of responses."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(inst.generate, prompt, image)
                for inst in self.instances
            ]
            return [f.result() for f in futures]
    
    def _merge_votes(self, responses: List[str], vote_type: str) -> Any:
        """Merge council responses via voting or intersection."""

@dataclass
class CompletionStatus:
    is_complete: bool
    quality_score: float  # 0.0 - 1.0
    missing_aspects: List[str]  # What's still needed
    reasoning: str  # Explanation for human debugging
    votes: Tuple[bool, bool, bool]  # Individual council votes

@dataclass
class CheckpointResult:
    approved: bool
    feedback: Optional[str]  # Guidance for iteration if not approved
    quality_score: float
    votes: Tuple[bool, bool, bool]  # Individual council votes
```

**VRAM Comparison:**
| Router Type | VRAM | Remaining for Experts |
|-------------|------|----------------------|
| Nemotron 30B (Ollama) | ~29GB | ~35GB |
| Council 3× Qwen3-VL:4B | ~27GB | ~37GB |

> **Note**: Each Qwen3-VL:4B instance needs ~9GB VRAM. Council uses 3 separate model instances for true parallel inference.

### 3.6 Council Mode (Temperature-Diverse)

**File**: `src/council_v7.py`

**Purpose**: Run N copies of same expert with varied temperatures for creative/exploratory tasks.

**Interface**:
```python
class CouncilMode:
    def __init__(self, slot_manager: ExpertSlotManager, critic_expert: str = None):
        """Initialize council with optional critic for selection."""
    
    def run(
        self, 
        query: str, 
        expert_id: str, 
        num_members: int = 5,
        temperatures: List[float] = [0.3, 0.5, 0.7, 0.9, 1.1]
    ) -> CouncilResult:
        """Run temperature-diverse council. Returns best response."""
    
    def select_best(self, responses: List[str], method: Literal["vote", "critic", "blend"]) -> str:
        """Select or synthesize best response."""
```

---

## 4. Execution Flow (Dual-Loop Implementation)

### 4.1 Main Orchestration (Outer Loop)

```python
def execute_task(query: str) -> str:
    """Main entry point - implements OUTER LOOP."""
    
    # Initial decomposition
    chunks = router.decompose(query)
    assignments = router.assign_chunks(chunks)
    
    outer_iteration = 0
    max_outer_iterations = 5  # Ralph loop limit
    
    while outer_iteration < max_outer_iterations:
        outer_iteration += 1
        
        # Load required experts
        for chunk_id, expert_id in assignments.items():
            slot_manager.load_expert(expert_id)
        
        # Execute all chunks (parallel, each runs INNER LOOP)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(execute_chunk, chunk, expert): chunk
                for chunk, expert in assignments.items()
            }
            wait(futures)  # All inner loops complete
        
        # OUTER LOOP DECISION POINT
        committed_work = memory.query_committed(query)
        completion_status = router.evaluate_completion(query, committed_work)
        
        if completion_status.is_complete:
            # Task done - synthesize and exit
            return router.synthesize(committed_work)
        else:
            # Task incomplete - re-decompose for gaps
            gap_chunks = router.identify_gaps(completion_status)
            assignments = router.assign_chunks(gap_chunks)
            # Continue outer loop
    
    # Max iterations reached - synthesize best effort
    return router.synthesize(memory.query_committed(query))
```

### 4.2 Chunk Execution (Inner Loop)

```python
def execute_chunk(chunk: TaskChunk, expert: ExpertInstance) -> None:
    """Execute single chunk - implements INNER LOOP."""
    
    inner_iteration = 0
    max_inner_iterations = 3  # Chunk refinement limit
    
    while inner_iteration < max_inner_iterations:
        inner_iteration += 1
        
        # Step 1: Expert generates output
        wip_id = expert.generate(chunk, context=memory.get_relevant_context())
        memory.add_wip(expert.id, wip_id)
        
        # Step 2: Crosstalk with peers
        crosstalk_messages = crosstalk_bus.receive(expert.id, timeout=3.0)
        for msg in crosstalk_messages:
            if msg.type == "request":
                response = expert.respond_to_peer(msg)
                crosstalk_bus.send(response)
        
        # Optionally request input from other experts
        if expert.needs_peer_input():
            request = expert.create_peer_request()
            crosstalk_bus.send(request)
            peer_response = crosstalk_bus.receive(expert.id, timeout=5.0)
            expert.incorporate_peer_input(peer_response)
        
        # Step 3: Router checkpoint (periodic, not every iteration)
        if inner_iteration % checkpoint_interval == 0:
            checkpoint_result = router.checkpoint(expert.id, wip_id)
            
            if checkpoint_result.approved:
                # INNER LOOP EXIT - chunk approved
                memory.commit(wip_id)
                return
            else:
                # Continue refining
                expert.apply_feedback(checkpoint_result.feedback)
    
    # Max inner iterations - commit best effort
    memory.commit(wip_id)
```

### 4.3 Council Sub-Flow

```
1. ROUTER detects creative/exploratory task (or user forces /council)
2. COUNCIL.run(query, expert_id, N, temperatures)
   └─ Load N copies of expert
   └─ Parallel generate at different temperatures
   └─ COUNCIL.select_best() via critic or voting
3. Best response → MEMORY.commit()
4. Unload copies, return slot to pool
```

### 4.4 Loop Termination Conditions

| Loop | Condition | Action |
|------|-----------|--------|
| **Outer** | `router.is_complete() == True` | Synthesize, exit |
| **Outer** | `outer_iteration >= max_outer` | Synthesize best-effort, exit |
| **Inner** | `checkpoint.approved == True` | Commit chunk, exit inner |
| **Inner** | `inner_iteration >= max_inner` | Commit best-effort, exit inner |
| **Inner** | `expert.timeout` | Escalate to Router, reassign |

---

## 5. Configuration

**File**: `config/coe_v7.yaml`

```yaml
system:
  max_vram_mb: 64000
  num_slots: 10
  router_model: "models/router_4b"

timing:
  crosstalk_interval_sec: 3
  checkpoint_interval_sec: 10
  expert_timeout_sec: 60

council:
  default_parallelism: 5
  temperatures: [0.3, 0.5, 0.7, 0.9, 1.1]
  selection_method: "critic"  # vote, critic, blend

memory:
  backend: "memvid"  # memvid, json, sqlite
  max_wip_items: 100
  commit_requires_approval: true

expert_library:
  catalog_path: "models/experts/catalog.json"
  preload_experts: ["python_expert", "general_reasoner"]
```

---

## 6. Implementation TODO List

### Phase 1: Core Infrastructure ✅
- [x] 1.1 Create `src/expert_slots.py` - ExpertSlotManager
- [x] 1.2 Create `src/expert_catalog.py` - Catalog loader
- [x] 1.3 Create `config/expert_catalog.json` - Initial catalog
- [ ] 1.4 Unit tests for slot manager

### Phase 2: Communication Layer ✅
- [x] 2.1 Create `src/crosstalk.py` - CrosstalkBus
- [x] 2.2 Create `src/episodic_memory.py` - EpisodicMemory
- [ ] 2.3 Integration tests for memory + crosstalk

### Phase 3: Council Router ✅
- [x] 3.1 Create `src/router_v7.py` - CouncilRouter
- [x] 3.2 Implement task decomposition with parallel inference
- [x] 3.3 Implement checkpoint logic with voting
- [x] 3.4 Implement completion evaluation (outer loop)
- [ ] 3.5 Router integration tests

### Phase 4: Council Mode ✅
- [x] 4.1 Create `src/council_v7.py` - CouncilMode
- [x] 4.2 Implement temperature-diverse generation
- [x] 4.3 Implement selection methods (vote, critic, blend)
- [ ] 4.4 Council integration tests

### Phase 5: Demo Application ✅
- [x] 5.1 Create `demo_coe_v7.py` - Main demo harness
- [x] 5.2 Implement CLI with /panel, /council, /status, /memory commands
- [x] 5.3 Implement session management
- [ ] 5.4 End-to-end testing

### Phase 6: Polish & Documentation
- [x] 6.1 Create `config/coe_v7.yaml` - Configuration
- [ ] 6.2 Error handling and recovery
- [ ] 6.3 Performance profiling
- [ ] 6.4 User documentation

---

## 7. Testing Protocol

### 7.1 Unit Tests

| Component | Test File | Key Tests |
|-----------|-----------|-----------|
| ExpertSlotManager | `tests/test_slots.py` | Load, unload, eviction, VRAM tracking |
| CrosstalkBus | `tests/test_crosstalk.py` | Send, receive, broadcast, timeout |
| EpisodicMemory | `tests/test_memory.py` | Add WIP, commit, query, rollback |
| RouterV7 | `tests/test_router.py` | Decompose, assign, checkpoint, complete |
| CouncilMode | `tests/test_council.py` | Run, select_best (all methods) |

### 7.2 Integration Tests

| Scenario | Description | Success Criteria |
|----------|-------------|------------------|
| Panel Basic | 3 different experts, simple query | All experts respond, synthesis generated |
| Council Creative | 5× same expert, varied temps | Best response selected, quality > single |
| HRM Full | Complex query, multiple rounds | Chunks assigned, crosstalk occurs, commits happen |
| Memory Integrity | WIP → Commit → Query | Committed work retrievable, WIP isolated |
| Dynamic Loading | Swap experts mid-task | No VRAM leak, new expert responds correctly |

### 7.3 End-to-End Tests

| Test | Query | Expected Behavior |
|------|-------|-------------------|
| E2E-1 | "Secure a Python FastAPI backend" | Router decomposes → Python + Security + Architecture → Synthesis |
| E2E-2 | "Write a mystery novel opening" | Council mode activates → 5 temps → Best selected |
| E2E-3 | "Analyze 5 Python files" | Multiple Python experts → Parallel analysis → Merged report |
| E2E-4 | Long task (>60s) | Multiple crosstalk + checkpoints observed |

### 7.4 Completion Criteria

#### Must Pass (Blocking)
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] E2E-1 through E2E-4 produce valid, coherent output
- [ ] No VRAM leaks after 10 consecutive queries
- [ ] Graceful shutdown (Ctrl+C) with cleanup

#### Should Pass (Quality Gates)
- [ ] Crosstalk logging shows inter-expert communication
- [ ] Episodic memory shows committed vs WIP items
- [ ] Council mode outperforms single-shot on E2E-2
- [ ] Dynamic expert swap completes in <5s (on NVMe)

#### Nice to Have
- [ ] Performance: <3s expert load time
- [ ] Performance: <30s for E2E-1 full synthesis
- [ ] Logging: Session replay from log file

---

## 8. File Structure

```
college of experts/
├── config/
│   └── coe_v7.yaml
├── docs/
│   └── COE_V7_SPECIFICATION.md   (this file)
├── models/
│   └── experts/
│       ├── catalog.json
│       └── [expert models...]
├── src/
│   ├── expert_slots.py
│   ├── expert_catalog.py
│   ├── crosstalk.py
│   ├── episodic_memory.py
│   ├── router_v7.py
│   ├── council_v7.py
│   └── ... (existing src files)
├── tests/
│   ├── test_slots.py
│   ├── test_crosstalk.py
│   ├── test_memory.py
│   ├── test_router.py
│   └── test_council.py
├── demo_coe_v7.py
└── demo_multi_expert_v6.py (legacy)
```

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| VRAM exhaustion | Strict budget tracking, aggressive eviction |
| Crosstalk deadlock | Timeouts on all message waits |
| Infinite iteration loop | Max checkpoint count per task |
| Expert load failure | Fallback to general_reasoner |
| Memory corruption | Committed partition is append-only |

---

## 10. Approval Checklist

- [ ] Architecture approved
- [ ] TODO phases approved
- [ ] Testing protocol approved
- [ ] Configuration schema approved
- [ ] File structure approved

**Awaiting your review and approval before proceeding with implementation.**
