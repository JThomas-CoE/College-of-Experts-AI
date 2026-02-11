"""
DAG Scheduler - Manages parallel slot execution based on dependency DAG.
"""

from typing import List, Dict, Set
from src.framework_scheduler import FrameworkSlot


class DAGScheduler:
    """
    Manages parallel slot execution based on dependency DAG.
    Executes independent slots concurrently up to VRAM budget.
    """
    
    def __init__(self, slots: List[FrameworkSlot], max_concurrent: int = 3):
        self.slots = {s.id: s for s in slots}
        self.slot_order = [s.id for s in slots]
        self.max_concurrent = max_concurrent
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.in_progress: Set[str] = set()
        
    def get_ready_slots(self) -> List[str]:
        """Get slots whose dependencies are all satisfied."""
        ready = []
        for slot_id in self.slot_order:
            if slot_id in self.completed or slot_id in self.failed or slot_id in self.in_progress:
                continue
            slot = self.slots[slot_id]
            deps_satisfied = all(d in self.completed for d in slot.dependencies)
            if deps_satisfied:
                ready.append(slot_id)
        return ready
    
    def can_start_more(self) -> bool:
        """Check if we can start more slots (within concurrency limit)."""
        return len(self.in_progress) < self.max_concurrent
    
    def start_slot(self, slot_id: str):
        """Mark a slot as in-progress."""
        self.in_progress.add(slot_id)
    
    def complete_slot(self, slot_id: str, success: bool):
        """Mark a slot as completed or failed."""
        self.in_progress.discard(slot_id)
        if success:
            self.completed.add(slot_id)
        else:
            self.failed.add(slot_id)
    
    def is_done(self) -> bool:
        """Check if all slots are processed."""
        return len(self.completed) + len(self.failed) == len(self.slots)
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        Analyze DAG and return groups that can run in parallel.
        
        Returns:
            List of slot ID groups, where each group can execute in parallel
        """
        groups = []
        temp_completed: Set[str] = set()
        
        while len(temp_completed) < len(self.slots):
            group = []
            for slot_id in self.slot_order:
                if slot_id in temp_completed:
                    continue
                slot = self.slots[slot_id]
                if all(d in temp_completed for d in slot.dependencies):
                    group.append(slot_id)
            if group:
                groups.append(group)
                temp_completed.update(group)
            else:
                break  # Circular dependency or error
        
        return groups
    
    def get_remaining_slots(self) -> List[str]:
        """Get slots that haven't completed or failed yet."""
        return [
            s for s in self.slot_order 
            if s not in self.completed and s not in self.failed
        ]