"""
Episodic Memory - Committed vs WIP Work Tracking

Part of College of Experts V7

Maintains two partitions:
- COMMITTED: Approved work, immutable, queryable by all experts
- WIP (Work-in-Progress): Mutable, owned by specific expert until commit
"""

import json
import os
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum
import hashlib


class MemoryStatus(Enum):
    WIP = "wip"
    PENDING_APPROVAL = "pending_approval"
    COMMITTED = "committed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class MemoryItem:
    """A single item in episodic memory."""
    id: str
    expert_id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.WIP
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    committed_at: Optional[datetime] = None
    version: int = 1
    parent_id: Optional[str] = None  # For tracking revisions
    chunk_ref: Optional[str] = None  # Reference to task chunk
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "expert_id": self.expert_id,
            "content": self.content,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "committed_at": self.committed_at.isoformat() if self.committed_at else None,
            "version": self.version,
            "parent_id": self.parent_id,
            "chunk_ref": self.chunk_ref
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        """Create from dict."""
        return cls(
            id=data["id"],
            expert_id=data["expert_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            status=MemoryStatus(data.get("status", "wip")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            committed_at=datetime.fromisoformat(data["committed_at"]) if data.get("committed_at") else None,
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            chunk_ref=data.get("chunk_ref")
        )


class EpisodicMemory:
    """
    Manages episodic memory with committed and WIP partitions.
    
    Features:
    - WIP items are mutable by owning expert
    - Committed items are immutable (append-only log)
    - Version tracking for revisions
    - Simple text search for queries
    
    Backends:
    - json: File-based JSON (default, good for debugging)
    - memvid: Video-based storage (future)
    - sqlite: SQLite database (future)
    """
    
    def __init__(
        self,
        storage_path: str = "data/episodic_memory",
        backend: Literal["json", "memvid", "sqlite"] = "json"
    ):
        self.storage_path = Path(storage_path)
        self.backend = backend
        
        # In-memory storage
        self._items: Dict[str, MemoryItem] = {}
        self._lock = threading.RLock()
        self._item_counter = 0
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()
    
    def _generate_id(self, prefix: str = "mem") -> str:
        """Generate unique item ID."""
        with self._lock:
            self._item_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"{prefix}_{timestamp}_{self._item_counter:04d}"
    
    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    # --- WIP Operations ---
    
    def add_wip(
        self,
        expert_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        chunk_ref: Optional[str] = None
    ) -> str:
        """
        Add a work-in-progress item.
        
        Args:
            expert_id: The expert creating this work
            content: The work content
            metadata: Optional metadata dict
            chunk_ref: Optional reference to task chunk
        
        Returns:
            Item ID
        """
        item_id = self._generate_id("wip")
        
        item = MemoryItem(
            id=item_id,
            expert_id=expert_id,
            content=content,
            metadata=metadata or {},
            status=MemoryStatus.WIP,
            chunk_ref=chunk_ref
        )
        
        with self._lock:
            self._items[item_id] = item
            self._save_to_disk()
        
        return item_id
    
    def update_wip(
        self,
        item_id: str,
        content: str,
        expert_id: str
    ) -> bool:
        """
        Update a WIP item (only owning expert can update).
        
        Returns:
            True if updated, False if not found or not authorized
        """
        with self._lock:
            item = self._items.get(item_id)
            
            if not item:
                return False
            
            if item.expert_id != expert_id:
                return False
            
            if item.status != MemoryStatus.WIP:
                return False
            
            # Create new version
            item.content = content
            item.version += 1
            item.updated_at = datetime.now()
            
            self._save_to_disk()
            return True
    
    def get_wip(self, expert_id: str) -> List[MemoryItem]:
        """Get all WIP items for a specific expert."""
        with self._lock:
            return [
                item for item in self._items.values()
                if item.expert_id == expert_id and item.status == MemoryStatus.WIP
            ]
    
    def submit_for_approval(self, item_id: str) -> bool:
        """Submit WIP item for Router approval."""
        with self._lock:
            item = self._items.get(item_id)
            if item and item.status == MemoryStatus.WIP:
                item.status = MemoryStatus.PENDING_APPROVAL
                item.updated_at = datetime.now()
                self._save_to_disk()
                return True
            return False
    
    # --- Commit Operations ---
    
    def commit(self, item_id: str, approver: str = "router") -> bool:
        """
        Commit a WIP item (move to committed partition).
        
        This operation is typically called by the Router after approval.
        Once committed, the item becomes immutable.
        
        Returns:
            True if committed successfully
        """
        with self._lock:
            item = self._items.get(item_id)
            
            if not item:
                return False
            
            if item.status not in [MemoryStatus.WIP, MemoryStatus.PENDING_APPROVAL]:
                return False
            
            item.status = MemoryStatus.COMMITTED
            item.committed_at = datetime.now()
            item.updated_at = datetime.now()
            item.metadata["approved_by"] = approver
            
            self._save_to_disk()
            return True
    
    def reject(self, item_id: str, reason: str = "") -> bool:
        """
        Reject a pending item.
        
        Returns:
            True if rejected successfully
        """
        with self._lock:
            item = self._items.get(item_id)
            
            if not item:
                return False
            
            item.status = MemoryStatus.REJECTED
            item.updated_at = datetime.now()
            item.metadata["rejection_reason"] = reason
            
            self._save_to_disk()
            return True
    
    def rollback(self, item_id: str) -> bool:
        """
        Rollback/discard a WIP item.
        
        Returns:
            True if rolled back
        """
        with self._lock:
            item = self._items.get(item_id)
            
            if not item:
                return False
            
            if item.status == MemoryStatus.COMMITTED:
                return False  # Cannot rollback committed items
            
            item.status = MemoryStatus.ROLLED_BACK
            item.updated_at = datetime.now()
            
            self._save_to_disk()
            return True
    
    # --- Query Operations ---
    
    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """Get a specific item by ID."""
        return self._items.get(item_id)
    
    def query_committed(
        self,
        query: Optional[str] = None,
        expert_id: Optional[str] = None,
        chunk_ref: Optional[str] = None,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """
        Query committed items.
        
        Args:
            query: Text to search for (simple substring match)
            expert_id: Filter by expert
            chunk_ref: Filter by chunk reference
            top_k: Maximum items to return
        
        Returns:
            List of matching committed items
        """
        with self._lock:
            items = [
                item for item in self._items.values()
                if item.status == MemoryStatus.COMMITTED
            ]
            
            if expert_id:
                items = [i for i in items if i.expert_id == expert_id]
            
            if chunk_ref:
                items = [i for i in items if i.chunk_ref == chunk_ref]
            
            if query:
                query_lower = query.lower()
                items = [i for i in items if query_lower in i.content.lower()]
            
            # Sort by commit time (newest first)
            items.sort(key=lambda x: x.committed_at or x.created_at, reverse=True)
            
            return items[:top_k]
    
    def get_all_committed(self) -> List[MemoryItem]:
        """Get all committed items."""
        return self.query_committed(top_k=1000)
    
    def get_pending_approval(self) -> List[MemoryItem]:
        """Get all items pending approval."""
        with self._lock:
            return [
                item for item in self._items.values()
                if item.status == MemoryStatus.PENDING_APPROVAL
            ]
    
    # --- Statistics ---
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            by_status = {}
            for item in self._items.values():
                status = item.status.value
                by_status[status] = by_status.get(status, 0) + 1
            
            return {
                "total_items": len(self._items),
                "by_status": by_status,
                "storage_path": str(self.storage_path),
                "backend": self.backend
            }
    
    # --- Persistence ---
    
    def _save_to_disk(self) -> None:
        """Save memory to disk (JSON backend)."""
        if self.backend != "json":
            return
        
        filepath = self.storage_path / "memory.json"
        data = {
            "counter": self._item_counter,
            "items": {k: v.to_dict() for k, v in self._items.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_from_disk(self) -> None:
        """Load memory from disk (JSON backend)."""
        if self.backend != "json":
            return
        
        filepath = self.storage_path / "memory.json"
        
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._item_counter = data.get("counter", 0)
            self._items = {
                k: MemoryItem.from_dict(v) 
                for k, v in data.get("items", {}).items()
            }
        except Exception as e:
            print(f"[EpisodicMemory] Warning: Could not load from disk: {e}")
    
    def clear(self) -> None:
        """Clear all memory (dangerous - use for testing)."""
        with self._lock:
            self._items.clear()
            self._item_counter = 0
            self._save_to_disk()
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"EpisodicMemory({stats['total_items']} items, backend={self.backend})"


# Convenience function
def create_memory(
    storage_path: str = "data/episodic_memory",
    backend: str = "json"
) -> EpisodicMemory:
    """Create an episodic memory instance."""
    return EpisodicMemory(storage_path, backend)


if __name__ == "__main__":
    # Quick test
    memory = EpisodicMemory(storage_path="data/test_memory")
    
    # Clear for clean test
    memory.clear()
    
    # Add WIP
    wip_id = memory.add_wip(
        expert_id="python_expert",
        content="def secure_hash(password): return bcrypt.hash(password)",
        metadata={"task": "password hashing"}
    )
    print(f"Added WIP: {wip_id}")
    
    # Query WIP
    wip_items = memory.get_wip("python_expert")
    print(f"Python expert WIP: {len(wip_items)} items")
    
    # Commit
    memory.commit(wip_id)
    print(f"Committed: {wip_id}")
    
    # Query committed
    committed = memory.query_committed(query="password")
    print(f"Committed items matching 'password': {len(committed)}")
    
    print(f"\nMemory state: {memory}")
    print(f"Stats: {memory.get_stats()}")
