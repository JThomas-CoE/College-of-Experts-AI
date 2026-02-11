"""
Memvid Memory Backbone - Video-encoded portable memory layer.

Uses Memvid to encode memories as QR codes in video format for:
- Extreme portability (single .mp4 file per memory tier)
- 10x compression vs traditional vector DBs
- Serverless operation (no database infrastructure)
- Expert-specific memory packages
"""

import time
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import hashlib

try:
    from memvid import MemvidEncoder, MemvidRetriever
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False
    MemvidEncoder = None
    MemvidRetriever = None


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    tier: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tier": self.tier,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            content=data["content"],
            tier=data["tier"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


@dataclass
class MemvidConfig:
    """Configuration for Memvid memory backbone."""
    memory_dir: Path = Path("data/memory")
    working_memory_max: int = 100  # Max entries before flush
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512  # Characters per chunk


class MemvidMemoryBackbone:
    """
    Memory backbone using Memvid video-encoded storage.
    
    Memory Tiers:
    - Working: In-memory dict (current session, fast access)
    - Episodic: Video-encoded past sessions (episodic_memory.mp4)
    - Semantic: Video-encoded facts/preferences (semantic_memory.mp4)
    - Expert-specific: Per-expert knowledge bases (experts/{id}/memory.mp4)
    """
    
    TIERS = ("working", "episodic", "semantic")
    
    def __init__(self, config: Optional[MemvidConfig] = None):
        self.config = config or MemvidConfig()
        self.config.memory_dir.mkdir(parents=True, exist_ok=True)
        
        if not MEMVID_AVAILABLE:
            print("Warning: memvid not installed. Using fallback mode.")
            print("Install with: pip install memvid")
        
        # L1: Working memory (in-memory, current session)
        self.working: Dict[str, MemoryEntry] = {}
        
        # L2/L3: Persistent memories (loaded on demand)
        self._episodic: Optional[Any] = None  # MemvidRetriever
        self._semantic: Optional[Any] = None  # MemvidRetriever
        
        # Expert-specific memories
        self._expert_memories: Dict[str, Any] = {}  # expert_id -> MemvidRetriever
        
        # Pending writes (batched for efficiency)
        self._pending_episodic: List[MemoryEntry] = []
        self._pending_semantic: List[MemoryEntry] = []
    
    @property
    def episodic(self) -> Optional[Any]:
        """Lazy-load episodic memory retriever."""
        if self._episodic is None and MEMVID_AVAILABLE:
            video_path = self.config.memory_dir / "episodic_memory.mp4"
            index_path = self.config.memory_dir / "episodic_memory.json"
            if video_path.exists() and index_path.exists():
                self._episodic = MemvidRetriever(str(video_path), str(index_path))
        return self._episodic
    
    @property
    def semantic(self) -> Optional[Any]:
        """Lazy-load semantic memory retriever."""
        if self._semantic is None and MEMVID_AVAILABLE:
            video_path = self.config.memory_dir / "semantic_memory.mp4"
            index_path = self.config.memory_dir / "semantic_memory.json"
            if video_path.exists() and index_path.exists():
                self._semantic = MemvidRetriever(str(video_path), str(index_path))
        return self._semantic
    
    def write(
        self,
        key: str,
        content: str,
        tier: str = "working",
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """
        Write content to memory.
        
        For working memory: immediate storage in dict
        For episodic/semantic: batched for later commit to video
        """
        entry = MemoryEntry(
            content=content,
            tier=tier,
            timestamp=time.time(),
            metadata=metadata or {"key": key}
        )
        entry.metadata["key"] = key
        
        if tier == "working":
            self.working[key] = entry
        elif tier == "episodic":
            self._pending_episodic.append(entry)
        elif tier == "semantic":
            self._pending_semantic.append(entry)
        else:
            raise ValueError(f"Unknown tier: {tier}")
        
        return entry
    
    def read(self, key: str, tier: str = "working") -> Optional[MemoryEntry]:
        """Read a specific memory by key."""
        if tier == "working":
            return self.working.get(key)
        
        # For video-stored memories, search by key
        results = self.recall(f"key:{key}", tiers=[tier], top_k=1)
        return results[0] if results else None
    
    def recall(
        self,
        query: str,
        tiers: Optional[List[str]] = None,
        top_k: int = 5,
        expert_id: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using semantic search.
        
        Args:
            query: Search query
            tiers: Which memory tiers to search (default: all)
            top_k: Maximum results per tier
            expert_id: Optional expert-specific memory to include
            
        Returns:
            List of relevant MemoryEntry objects
        """
        if tiers is None:
            tiers = list(self.TIERS)
        
        results = []
        
        for tier in tiers:
            if tier == "working":
                # Search working memory (simple text match for now)
                for entry in self.working.values():
                    if query.lower() in entry.content.lower():
                        results.append(entry)
            
            elif tier == "episodic" and self.episodic and MEMVID_AVAILABLE:
                try:
                    tier_results = self.episodic.search(query, top_k=top_k)
                    for text in tier_results:
                        results.append(MemoryEntry(
                            content=text,
                            tier="episodic",
                            timestamp=time.time(),
                            metadata={"source": "memvid_search"}
                        ))
                except Exception as e:
                    print(f"Episodic search error: {e}")
            
            elif tier == "semantic" and self.semantic and MEMVID_AVAILABLE:
                try:
                    tier_results = self.semantic.search(query, top_k=top_k)
                    for text in tier_results:
                        results.append(MemoryEntry(
                            content=text,
                            tier="semantic",
                            timestamp=time.time(),
                            metadata={"source": "memvid_search"}
                        ))
                except Exception as e:
                    print(f"Semantic search error: {e}")
        
        # Also search expert-specific memory if requested
        if expert_id and expert_id in self._expert_memories:
            try:
                expert_results = self._expert_memories[expert_id].search(query, top_k=top_k)
                for text in expert_results:
                    results.append(MemoryEntry(
                        content=text,
                        tier=f"expert:{expert_id}",
                        timestamp=time.time(),
                        metadata={"expert_id": expert_id}
                    ))
            except Exception as e:
                print(f"Expert memory search error: {e}")
        
        return results[:top_k * len(tiers)]  # Limit total results
    
    def load_expert_memory(self, expert_id: str) -> bool:
        """
        Load expert-specific memory when expert is activated.
        
        Expert memories are stored in: experts/{expert_id}/memory.mp4
        """
        if not MEMVID_AVAILABLE:
            return False
        
        if expert_id in self._expert_memories:
            return True
        
        expert_dir = self.config.memory_dir / "experts" / expert_id
        video_path = expert_dir / "memory.mp4"
        index_path = expert_dir / "memory.json"
        
        if video_path.exists() and index_path.exists():
            try:
                self._expert_memories[expert_id] = MemvidRetriever(
                    str(video_path), str(index_path)
                )
                return True
            except Exception as e:
                print(f"Failed to load expert memory for {expert_id}: {e}")
        
        return False
    
    def unload_expert_memory(self, expert_id: str):
        """Unload expert memory to free resources."""
        self._expert_memories.pop(expert_id, None)
    
    def commit(self):
        """
        Commit pending memories to video storage.
        
        This encodes accumulated entries into the appropriate .mp4 files.
        Call periodically or at session end.
        """
        if not MEMVID_AVAILABLE:
            print("Memvid not available, skipping commit")
            return
        
        if self._pending_episodic:
            self._commit_tier("episodic", self._pending_episodic)
            self._pending_episodic = []
            self._episodic = None  # Force reload on next access
        
        if self._pending_semantic:
            self._commit_tier("semantic", self._pending_semantic)
            self._pending_semantic = []
            self._semantic = None  # Force reload on next access
    
    def _commit_tier(self, tier: str, entries: List[MemoryEntry]):
        """Encode entries into video for a specific tier."""
        if not entries:
            return
        
        video_path = self.config.memory_dir / f"{tier}_memory.mp4"
        index_path = self.config.memory_dir / f"{tier}_memory.json"
        
        # Create encoder and add entries
        encoder = MemvidEncoder()
        
        for entry in entries:
            # Format entry as searchable text
            text = self._format_entry_for_storage(entry)
            encoder.add_text(text)
        
        # Build/rebuild video
        encoder.build_video(str(video_path), str(index_path))
        print(f"Committed {len(entries)} entries to {tier} memory")
    
    def _format_entry_for_storage(self, entry: MemoryEntry) -> str:
        """Format a memory entry for video storage."""
        meta_str = json.dumps(entry.metadata) if entry.metadata else "{}"
        return f"[{entry.timestamp}] {entry.content}\n---metadata: {meta_str}"
    
    def synthesize_context(
        self,
        query: str,
        max_tokens: int = 2000,
        expert_id: Optional[str] = None
    ) -> str:
        """
        Synthesize relevant context from all memory tiers.
        
        This is the main interface for injecting memory into prompts.
        """
        # Gather relevant memories
        memories = self.recall(
            query,
            tiers=["working", "episodic", "semantic"],
            top_k=5,
            expert_id=expert_id
        )
        
        if not memories:
            return ""
        
        # Build context string
        context_parts = ["[MEMORY CONTEXT]"]
        
        # Group by tier
        by_tier: Dict[str, List[MemoryEntry]] = {}
        for mem in memories:
            tier = mem.tier
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(mem)
        
        for tier, tier_memories in by_tier.items():
            context_parts.append(f"\n## {tier.title()} Memory:")
            for mem in tier_memories[:3]:  # Limit per tier
                content = mem.content[:500]  # Truncate long entries
                context_parts.append(f"- {content}")
        
        context_parts.append("[END MEMORY CONTEXT]")
        
        return "\n".join(context_parts)
    
    # Convenience methods
    
    def set_preference(self, key: str, value: Any):
        """Store a user preference in semantic memory."""
        content = f"User preference: {key} = {json.dumps(value)}"
        self.write(f"pref:{key}", content, tier="semantic", metadata={"type": "preference", "key": key})
    
    def log_session(self, session_id: str, summary: str):
        """Log a session summary to episodic memory."""
        content = f"Session {session_id}: {summary}"
        self.write(f"session:{session_id}", content, tier="episodic", metadata={"type": "session"})
    
    def get_context(self, key: str = "current") -> Dict[str, Any]:
        """Get current working context."""
        entry = self.working.get(f"context:{key}")
        if entry:
            try:
                return json.loads(entry.content)
            except json.JSONDecodeError:
                return {"raw": entry.content}
        return {}
    
    def update_context(self, updates: Dict[str, Any], key: str = "current"):
        """Update current working context."""
        current = self.get_context(key)
        current.update(updates)
        self.write(f"context:{key}", json.dumps(current), tier="working")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "working_count": len(self.working),
            "pending_episodic": len(self._pending_episodic),
            "pending_semantic": len(self._pending_semantic),
            "expert_memories_loaded": list(self._expert_memories.keys()),
            "memvid_available": MEMVID_AVAILABLE
        }
        
        # Check if video files exist
        for tier in ["episodic", "semantic"]:
            video_path = self.config.memory_dir / f"{tier}_memory.mp4"
            stats[f"{tier}_exists"] = video_path.exists()
        
        return stats
    
    def clear_working(self):
        """Clear working memory (end of session)."""
        self.working.clear()


# Fallback implementation when Memvid is not available
class FallbackMemoryBackbone(MemvidMemoryBackbone):
    """
    Fallback memory using simple JSON files when Memvid is unavailable.
    """
    
    def __init__(self, config: Optional[MemvidConfig] = None):
        super().__init__(config)
        self._episodic_file = self.config.memory_dir / "episodic_fallback.json"
        self._semantic_file = self.config.memory_dir / "semantic_fallback.json"
        self._load_fallback_data()
    
    def _load_fallback_data(self):
        """Load JSON fallback data."""
        self._episodic_data: List[Dict] = []
        self._semantic_data: List[Dict] = []
        
        if self._episodic_file.exists():
            with open(self._episodic_file) as f:
                self._episodic_data = json.load(f)
        
        if self._semantic_file.exists():
            with open(self._semantic_file) as f:
                self._semantic_data = json.load(f)
    
    def recall(
        self,
        query: str,
        tiers: Optional[List[str]] = None,
        top_k: int = 5,
        expert_id: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Simple text search fallback."""
        if tiers is None:
            tiers = list(self.TIERS)
        
        results = []
        query_lower = query.lower()
        
        for tier in tiers:
            if tier == "working":
                for entry in self.working.values():
                    if query_lower in entry.content.lower():
                        results.append(entry)
            elif tier == "episodic":
                for item in self._episodic_data:
                    if query_lower in item.get("content", "").lower():
                        results.append(MemoryEntry.from_dict(item))
            elif tier == "semantic":
                for item in self._semantic_data:
                    if query_lower in item.get("content", "").lower():
                        results.append(MemoryEntry.from_dict(item))
        
        return results[:top_k]
    
    def commit(self):
        """Commit to JSON files."""
        if self._pending_episodic:
            self._episodic_data.extend([e.to_dict() for e in self._pending_episodic])
            with open(self._episodic_file, "w") as f:
                json.dump(self._episodic_data, f, indent=2)
            self._pending_episodic = []
        
        if self._pending_semantic:
            self._semantic_data.extend([e.to_dict() for e in self._pending_semantic])
            with open(self._semantic_file, "w") as f:
                json.dump(self._semantic_data, f, indent=2)
            self._pending_semantic = []


def create_memory_backbone(config: Optional[MemvidConfig] = None) -> MemvidMemoryBackbone:
    """Factory function to create appropriate memory backbone."""
    if MEMVID_AVAILABLE:
        return MemvidMemoryBackbone(config)
    else:
        print("Memvid not available, using JSON fallback")
        return FallbackMemoryBackbone(config)


if __name__ == "__main__":
    # Quick test
    config = MemvidConfig(memory_dir=Path("test_memory"))
    memory = create_memory_backbone(config)
    
    # Test working memory
    memory.write("test_key", "This is a test memory", tier="working")
    result = memory.read("test_key", tier="working")
    print(f"Working memory test: {result}")
    
    # Test context synthesis
    context = memory.synthesize_context("test")
    print(f"Context: {context}")
    
    # Stats
    print(f"Stats: {memory.get_stats()}")
