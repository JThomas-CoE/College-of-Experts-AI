"""
Memory Backbone - Shared memory layer for inter-expert communication.

Implements three memory tiers:
1. Working Memory: Current task context and scratchpad
2. Episodic Memory: Past sessions and interaction patterns  
3. Semantic Memory: Learned facts and user preferences
"""

import json
import time
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional, Any, List
from pathlib import Path
from contextlib import contextmanager


@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    value: Any
    tier: str  # "working", "episodic", "semantic"
    created_at: float
    updated_at: float
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryConfig:
    """Configuration for the memory backbone."""
    db_path: Path = Path("memory.db")
    working_memory_ttl: int = 3600  # 1 hour
    max_working_entries: int = 1000
    max_episodic_entries: int = 10000


class MemoryBackbone:
    """
    Shared memory layer enabling inter-expert communication
    and persistent learning without model retraining.
    """
    
    TIERS = ("working", "episodic", "semantic")
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for memory storage."""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (key, tier)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memory(tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON memory(updated_at)")
    
    @contextmanager
    def _get_conn(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def write(
        self,
        key: str,
        value: Any,
        tier: str = "working",
        metadata: Optional[dict] = None
    ) -> MemoryEntry:
        """
        Write a value to memory.
        
        Args:
            key: Unique identifier for this memory
            value: The value to store (will be JSON serialized)
            tier: Memory tier ("working", "episodic", "semantic")
            metadata: Optional metadata dict
            
        Returns:
            The created MemoryEntry
        """
        if tier not in self.TIERS:
            raise ValueError(f"Invalid tier: {tier}. Must be one of {self.TIERS}")
        
        now = time.time()
        value_json = json.dumps(value)
        meta_json = json.dumps(metadata) if metadata else None
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO memory (key, tier, value, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key, tier) DO UPDATE SET
                    value = excluded.value,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
            """, (key, tier, value_json, meta_json, now, now))
        
        return MemoryEntry(
            key=key,
            value=value,
            tier=tier,
            created_at=now,
            updated_at=now,
            metadata=metadata
        )
    
    def read(self, key: str, tier: str = "working") -> Optional[MemoryEntry]:
        """
        Read a value from memory.
        
        Args:
            key: The key to look up
            tier: Memory tier to search
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM memory WHERE key = ? AND tier = ?",
                (key, tier)
            ).fetchone()
            
            if row:
                return MemoryEntry(
                    key=row["key"],
                    value=json.loads(row["value"]),
                    tier=row["tier"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None
                )
        return None
    
    def search(
        self,
        query: str,
        tier: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memory by key prefix or content.
        
        Args:
            query: Search query (prefix match on key)
            tier: Optional tier to limit search
            limit: Maximum results to return
            
        Returns:
            List of matching MemoryEntry objects
        """
        results = []
        
        with self._get_conn() as conn:
            if tier:
                rows = conn.execute(
                    "SELECT * FROM memory WHERE tier = ? AND key LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (tier, f"{query}%", limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (f"{query}%", limit)
                ).fetchall()
            
            for row in rows:
                results.append(MemoryEntry(
                    key=row["key"],
                    value=json.loads(row["value"]),
                    tier=row["tier"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None
                ))
        
        return results
    
    def delete(self, key: str, tier: str = "working") -> bool:
        """Delete a memory entry."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory WHERE key = ? AND tier = ?",
                (key, tier)
            )
            return cursor.rowcount > 0
    
    def clear_tier(self, tier: str):
        """Clear all entries in a tier."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM memory WHERE tier = ?", (tier,))
    
    def cleanup_working_memory(self):
        """Remove expired working memory entries."""
        cutoff = time.time() - self.config.working_memory_ttl
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM memory WHERE tier = 'working' AND updated_at < ?",
                (cutoff,)
            )
    
    def get_tier_stats(self) -> dict:
        """Get statistics for each memory tier."""
        stats = {}
        with self._get_conn() as conn:
            for tier in self.TIERS:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM memory WHERE tier = ?",
                    (tier,)
                ).fetchone()
                stats[tier] = row["count"]
        return stats
    
    # Convenience methods for common patterns
    
    def set_user_preference(self, key: str, value: Any):
        """Store a user preference in semantic memory."""
        self.write(f"user.preference.{key}", value, tier="semantic")
    
    def get_user_preference(self, key: str) -> Optional[Any]:
        """Retrieve a user preference."""
        entry = self.read(f"user.preference.{key}", tier="semantic")
        return entry.value if entry else None
    
    def log_session(self, session_id: str, summary: dict):
        """Log a session summary to episodic memory."""
        self.write(
            f"session.{session_id}",
            summary,
            tier="episodic",
            metadata={"type": "session_summary"}
        )
    
    def get_context(self, context_key: str = "current") -> dict:
        """Get the current working context."""
        entry = self.read(f"context.{context_key}", tier="working")
        return entry.value if entry else {}
    
    def update_context(self, updates: dict, context_key: str = "current"):
        """Update the current working context."""
        current = self.get_context(context_key)
        current.update(updates)
        self.write(f"context.{context_key}", current, tier="working")


if __name__ == "__main__":
    # Quick test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(db_path=Path(tmpdir) / "test_memory.db")
        memory = MemoryBackbone(config)
        
        # Test writes
        memory.write("task.current", {"goal": "Write Python code"}, tier="working")
        memory.set_user_preference("coding_style", "functional")
        
        # Test reads
        task = memory.read("task.current", tier="working")
        print(f"Current task: {task.value if task else 'None'}")
        
        pref = memory.get_user_preference("coding_style")
        print(f"Coding style preference: {pref}")
        
        # Test stats
        print(f"Memory stats: {memory.get_tier_stats()}")
