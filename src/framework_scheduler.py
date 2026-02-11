"""
V12 Framework Scheduler - DAG-based Task Execution
College of Experts Architecture

Handles:
- Task framework slot management
- Dependency resolution and parallel execution
- Status tracking and blocking detection
- Final output assembly
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import json


class SlotStatus(Enum):
    """Status of a framework slot."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class FrameworkSlot:
    """
    Individual task unit within a TaskFramework.
    """
    id: str
    title: str
    description: str
    expected_outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)      # Slot IDs that must complete first
    can_reference: List[str] = field(default_factory=list)     # Slot IDs available for read-only access
    reference_hints: Dict[str, List[str]] = field(default_factory=dict)  # Specific sections to reference
    status: SlotStatus = SlotStatus.PENDING
    assigned_expert: Optional[str] = None  # Primary persona for this slot
    persona: Optional[str] = None          # Alias for assigned_expert (persona-aware routing)
    result: Optional[Any] = None  # StructuredSlotResult when done
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        # Use persona if set, otherwise assigned_expert
        expert = self.persona or self.assigned_expert
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "expected_outputs": self.expected_outputs,
            "dependencies": self.dependencies,
            "can_reference": self.can_reference,
            "reference_hints": self.reference_hints,
            "status": self.status.value,
            "persona": expert,
            "assigned_expert": expert,
            "error": self.error,
            "attempts": self.attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FrameworkSlot":
        """Create from dictionary."""
        # Support both 'persona' and 'assigned_expert' fields
        persona = data.get("persona") or data.get("assigned_expert")
        slot = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            expected_outputs=data.get("expected_outputs", []),
            dependencies=data.get("dependencies", []),
            can_reference=data.get("can_reference", []),
            reference_hints=data.get("reference_hints", {}),
            persona=persona,
            assigned_expert=persona
        )
        if "status" in data:
            slot.status = SlotStatus(data["status"])
        return slot


@dataclass
class TaskFramework:
    """
    Structured task decomposition with slots and dependencies.
    """
    id: str
    title: str
    description: str
    slots: List[FrameworkSlot] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_slot(self, slot_id: str) -> Optional[FrameworkSlot]:
        """Get slot by ID."""
        for slot in self.slots:
            if slot.id == slot_id:
                return slot
        return None
    
    def add_slot(self, slot: FrameworkSlot):
        """Add a slot to the framework."""
        self.slots.append(slot)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "slots": [s.to_dict() for s in self.slots],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TaskFramework":
        """Create from dictionary."""
        framework = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            metadata=data.get("metadata", {})
        )
        for slot_data in data.get("slots", []):
            framework.add_slot(FrameworkSlot.from_dict(slot_data))
        return framework
    
    def validate(self) -> List[str]:
        """
        Validate framework structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        slot_ids = {s.id for s in self.slots}
        
        for slot in self.slots:
            # Check dependencies exist
            for dep_id in slot.dependencies:
                if dep_id not in slot_ids:
                    errors.append(f"Slot '{slot.id}' depends on unknown slot '{dep_id}'")
            
            # Check references exist
            for ref_id in slot.can_reference:
                if ref_id not in slot_ids:
                    errors.append(f"Slot '{slot.id}' references unknown slot '{ref_id}'")
            
            # Check for self-dependency
            if slot.id in slot.dependencies:
                errors.append(f"Slot '{slot.id}' depends on itself")
        
        # Check for circular dependencies
        if self._has_circular_deps():
            errors.append("Framework contains circular dependencies")
        
        return errors
    
    def _has_circular_deps(self) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(slot_id: str) -> bool:
            visited.add(slot_id)
            rec_stack.add(slot_id)
            
            slot = self.get_slot(slot_id)
            if slot:
                for dep_id in slot.dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(slot_id)
            return False
        
        for slot in self.slots:
            if slot.id not in visited:
                if dfs(slot.id):
                    return True
        
        return False


class FrameworkScheduler:
    """
    Executes a TaskFramework with dependency-aware scheduling.
    """
    
    def __init__(
        self,
        framework: TaskFramework,
        working_memory: Any,  # WorkingMemory instance
        expert_executor: Callable,  # async function(slot, context) -> result
        max_parallel: int = 4
    ):
        """
        Initialize scheduler.
        
        Args:
            framework: TaskFramework to execute
            working_memory: WorkingMemory instance for storing results
            expert_executor: Async function to execute a slot
            max_parallel: Maximum parallel slot executions
        """
        self.framework = framework
        self.memory = working_memory
        self.executor = expert_executor
        self.max_parallel = max_parallel
        self._execution_log: List[Dict] = []
    
    def get_ready_slots(self) -> List[FrameworkSlot]:
        """
        Get slots whose dependencies are satisfied.
        
        Returns:
            List of slots ready for execution
        """
        ready = []
        
        for slot in self.framework.slots:
            if slot.status != SlotStatus.PENDING:
                continue
            
            # Check all dependencies are done
            deps_satisfied = all(
                self.framework.get_slot(dep_id).status == SlotStatus.DONE
                for dep_id in slot.dependencies
                if self.framework.get_slot(dep_id)
            )
            
            if deps_satisfied:
                ready.append(slot)
        
        return ready
    
    def get_blocked_slots(self) -> List[FrameworkSlot]:
        """Get slots that are blocked by failed dependencies."""
        blocked = []
        
        for slot in self.framework.slots:
            if slot.status != SlotStatus.PENDING:
                continue
            
            # Check if any dependency failed
            for dep_id in slot.dependencies:
                dep_slot = self.framework.get_slot(dep_id)
                if dep_slot and dep_slot.status == SlotStatus.FAILED:
                    blocked.append(slot)
                    break
        
        return blocked
    
    def update_blocked_statuses(self):
        """Update status of slots blocked by failed dependencies."""
        for slot in self.get_blocked_slots():
            slot.status = SlotStatus.BLOCKED
            slot.error = "Blocked by failed dependency"
    
    def is_complete(self) -> bool:
        """Check if all slots are done, failed, or blocked."""
        for slot in self.framework.slots:
            if slot.status in (SlotStatus.PENDING, SlotStatus.IN_PROGRESS):
                return False
        return True
    
    def has_pending(self) -> bool:
        """Check if there are pending slots."""
        return any(s.status == SlotStatus.PENDING for s in self.framework.slots)
    
    def build_slot_context(self, slot: FrameworkSlot) -> Dict:
        """
        Build execution context for a slot.
        
        Returns:
            Dict with slot info and reference content
        """
        context = {
            "slot_id": slot.id,
            "title": slot.title,
            "description": slot.description,
            "expected_outputs": slot.expected_outputs,
            "assigned_expert": slot.assigned_expert,
            "attempt": slot.attempts + 1
        }
        
        # Build reference context
        references = {}
        for ref_id in slot.can_reference:
            if self.memory.has(ref_id):
                ref_config = {
                    "type": "outline"  # Default to outline
                }
                if ref_id in slot.reference_hints:
                    ref_config = {
                        "type": "sections",
                        "hints": slot.reference_hints[ref_id]
                    }
                references[ref_id] = ref_config
        
        if references:
            context["references"] = self.memory.build_reference_context(references)
        
        return context
    
    async def execute_slot(self, slot: FrameworkSlot) -> bool:
        """
        Execute a single slot.
        
        Returns:
            True if successful, False otherwise
        """
        slot.status = SlotStatus.IN_PROGRESS
        slot.started_at = datetime.now()
        slot.attempts += 1
        
        try:
            context = self.build_slot_context(slot)
            
            # Execute via provided executor
            result = await self.executor(slot, context)
            
            # Store result in working memory
            self.memory.store_parsed(result)
            
            slot.result = result
            slot.status = SlotStatus.DONE
            slot.completed_at = datetime.now()
            
            self._execution_log.append({
                "slot_id": slot.id,
                "status": "success",
                "attempt": slot.attempts,
                "duration_ms": (slot.completed_at - slot.started_at).total_seconds() * 1000
            })
            
            return True
            
        except Exception as e:
            slot.error = str(e)
            
            if slot.attempts >= slot.max_attempts:
                slot.status = SlotStatus.FAILED
                self._execution_log.append({
                    "slot_id": slot.id,
                    "status": "failed",
                    "attempt": slot.attempts,
                    "error": str(e)
                })
                return False
            else:
                # Allow retry
                slot.status = SlotStatus.PENDING
                self._execution_log.append({
                    "slot_id": slot.id,
                    "status": "retry",
                    "attempt": slot.attempts,
                    "error": str(e)
                })
                return False
    
    async def execute_framework(self) -> Dict:
        """
        Execute the entire framework with dependency-aware scheduling.
        
        Returns:
            Dict with execution results and statistics
        """
        start_time = datetime.now()
        iterations = 0
        max_iterations = len(self.framework.slots) * 3  # Safety limit
        
        while not self.is_complete() and iterations < max_iterations:
            iterations += 1
            
            # Get ready slots
            ready_slots = self.get_ready_slots()
            
            if not ready_slots:
                if self.has_pending():
                    # Deadlock or waiting for retries
                    self.update_blocked_statuses()
                    if not self.has_pending():
                        break
                    # Small delay before retry check
                    await asyncio.sleep(0.1)
                    continue
                break
            
            # Limit parallel execution
            batch = ready_slots[:self.max_parallel]
            
            # Execute batch in parallel
            await asyncio.gather(*[
                self.execute_slot(slot) for slot in batch
            ])
            
            # Update blocked statuses
            self.update_blocked_statuses()
        
        end_time = datetime.now()
        
        return {
            "framework_id": self.framework.id,
            "total_slots": len(self.framework.slots),
            "completed": sum(1 for s in self.framework.slots if s.status == SlotStatus.DONE),
            "failed": sum(1 for s in self.framework.slots if s.status == SlotStatus.FAILED),
            "blocked": sum(1 for s in self.framework.slots if s.status == SlotStatus.BLOCKED),
            "iterations": iterations,
            "duration_ms": (end_time - start_time).total_seconds() * 1000,
            "execution_log": self._execution_log
        }
    
    def assemble_final_output(self) -> str:
        """
        Assemble completed slots into final output.
        
        Returns:
            Formatted final output string
        """
        sections = []
        
        sections.append(f"# {self.framework.title}\n")
        sections.append(f"{self.framework.description}\n")
        sections.append("---\n")
        
        for slot in self.framework.slots:
            section = f"## {slot.title}\n\n"
            
            if slot.status == SlotStatus.DONE and slot.result:
                section += slot.result.raw_content
            elif slot.status == SlotStatus.FAILED:
                section += f"⚠️ **Failed**: {slot.error}\n\n"
                section += f"*Attempted scope*: {slot.description}"
            elif slot.status == SlotStatus.BLOCKED:
                section += f"⏸️ **Blocked**: {slot.error}\n\n"
                section += f"*Planned scope*: {slot.description}"
            else:
                section += f"⏳ **Not completed** (status: {slot.status.value})"
            
            sections.append(section)
        
        return "\n\n---\n\n".join(sections)
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        status_counts = {}
        for slot in self.framework.slots:
            status = slot.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        parts = [f"{status}: {count}" for status, count in status_counts.items()]
        return f"Framework '{self.framework.id}': " + ", ".join(parts)


# Utility function to create framework from decomposition
def create_framework_from_plan(
    framework_id: str,
    title: str,
    description: str,
    slots_config: List[Dict]
) -> TaskFramework:
    """
    Create a TaskFramework from a plan configuration.
    
    Args:
        framework_id: Unique identifier
        title: Framework title
        description: Framework description
        slots_config: List of slot configurations
        
    Returns:
        TaskFramework instance
    """
    framework = TaskFramework(
        id=framework_id,
        title=title,
        description=description
    )
    
    for config in slots_config:
        slot = FrameworkSlot(
            id=config["id"],
            title=config["title"],
            description=config["description"],
            expected_outputs=config.get("expected_outputs", []),
            dependencies=config.get("dependencies", []),
            can_reference=config.get("can_reference", []),
            reference_hints=config.get("reference_hints", {})
        )
        framework.add_slot(slot)
    
    # Validate
    errors = framework.validate()
    if errors:
        raise ValueError(f"Invalid framework: {errors}")
    
    return framework


if __name__ == "__main__":
    # Test framework scheduler
    print("Testing FrameworkScheduler...")
    
    # Create test framework
    framework = create_framework_from_plan(
        framework_id="test_hipaa",
        title="HIPAA-Compliant Medical Records System",
        description="Build secure medical records analysis system",
        slots_config=[
            {
                "id": "security_design",
                "title": "Security Architecture",
                "description": "Design HIPAA-compliant security architecture",
                "expected_outputs": ["Encryption approach", "Access control model"],
                "dependencies": [],
                "can_reference": []
            },
            {
                "id": "implementation",
                "title": "Python Implementation",
                "description": "Implement secure database access",
                "expected_outputs": ["Database connection code", "Encryption functions"],
                "dependencies": ["security_design"],
                "can_reference": ["security_design"],
                "reference_hints": {"security_design": ["1.1", "1.2"]}
            },
            {
                "id": "legal_disclaimer",
                "title": "Delaware Disclaimer",
                "description": "Draft Delaware jurisdiction disclaimer",
                "expected_outputs": ["Liability limitation", "No warranty clause"],
                "dependencies": [],
                "can_reference": []
            },
            {
                "id": "integration",
                "title": "Integration",
                "description": "Wire components together",
                "expected_outputs": ["Main script", "Test cases"],
                "dependencies": ["implementation", "legal_disclaimer"],
                "can_reference": ["implementation", "legal_disclaimer"]
            }
        ]
    )
    
    print(f"\nFramework: {framework.title}")
    print(f"Slots: {len(framework.slots)}")
    print(f"Validation errors: {framework.validate()}")
    
    print("\nSlot dependencies:")
    for slot in framework.slots:
        deps = slot.dependencies if slot.dependencies else "[none]"
        print(f"  {slot.id}: {deps}")
    
    print("\nSerialized framework:")
    print(json.dumps(framework.to_dict(), indent=2, default=str)[:500] + "...")
