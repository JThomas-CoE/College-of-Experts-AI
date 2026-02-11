"""
Memory Vector Router - Routes queries to WorkingMemory or Expert Generation
College of Experts Architecture

Handles:
- Vector similarity search against WorkingMemory contents
- Cache hit detection for redundant queries
- Fallback to expert routing when no memory match
- Integration with WorkingMemory (Tier 1) and VectorRouter
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .working_memory import WorkingMemory, StructuredSlotResult
from .vector_router import VectorRouter


@dataclass
class MemoryMatch:
    """Result of a memory lookup."""
    slot_id: str
    expert_id: str
    similarity: float
    content: str
    is_cached: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "slot_id": self.slot_id,
            "expert_id": self.expert_id,
            "similarity": self.similarity,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "is_cached": self.is_cached
        }


class MemoryVectorRouter:
    """
    Routes queries to either WorkingMemory (cached results) or expert generation.
    
    Uses vector similarity to determine if a similar query was already answered.
    If similarity > threshold, returns cached result from WorkingMemory.
    Otherwise, routes to VectorRouter for expert assignment.
    
    This implements the "memory-first routing" pattern for the 3-tier architecture.
    """
    
    def __init__(
        self,
        working_memory: WorkingMemory,
        vector_router: VectorRouter,
        embedding_manager,
        similarity_threshold: float = 0.85,
        min_content_length: int = 100
    ):
        """
        Initialize memory vector router.
        
        Args:
            working_memory: Tier 1 memory store for completed slots
            vector_router: Fallback router for expert assignment
            embedding_manager: Shared embedding manager for vector operations
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            min_content_length: Minimum content length to consider for caching
        """
        self.working_memory = working_memory
        self.vector_router = vector_router
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.min_content_length = min_content_length
        
        # Cache for query embeddings to avoid re-encoding
        self._query_cache: Dict[str, np.ndarray] = {}
        self._slot_embeddings: Dict[str, np.ndarray] = {}
        
        # Statistics for monitoring
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "expert_routes": 0,
            "total_queries": 0
        }
    
    def route(
        self,
        query: str,
        context: Optional[str] = None,
        slot_requirements: Optional[Dict] = None
    ) -> Dict:
        """
        Route a query to memory or expert.
        
        Args:
            query: User query to route
            context: Optional context for embedding
            slot_requirements: Optional requirements dict from framework
            
        Returns:
            Dict with routing decision:
            {
                "source": "memory" | "expert",
                "content": str,  # Cached content if memory hit
                "expert_id": str,  # If expert route
                "confidence": float,
                "match_info": Dict  # If memory hit
            }
        """
        self.stats["total_queries"] += 1
        
        # Step 1: Check WorkingMemory for similar past results
        memory_match = self._search_memory(query, context)
        
        if memory_match and memory_match.similarity >= self.similarity_threshold:
            # Memory hit - return cached result
            self.stats["memory_hits"] += 1
            return {
                "source": "memory",
                "content": memory_match.content,
                "expert_id": memory_match.expert_id,
                "confidence": memory_match.similarity,
                "match_info": {
                    "slot_id": memory_match.slot_id,
                    "similarity": memory_match.similarity,
                    "is_exact": memory_match.similarity > 0.95
                }
            }
        
        # Step 2: Memory miss - route to expert via VectorRouter
        self.stats["memory_misses"] += 1
        self.stats["expert_routes"] += 1
        
        # If slot_requirements specify expert directly, use it
        if slot_requirements and "expert_id" in slot_requirements:
            expert_id = slot_requirements["expert_id"]
            confidence = 1.0
        else:
            # Use VectorRouter for expert selection
            expert_id, confidence, _ = self.vector_router.route(query)
        
        return {
            "source": "expert",
            "expert_id": expert_id,
            "confidence": confidence,
            "memory_match": memory_match.to_dict() if memory_match else None,
            "reason": "No sufficient memory match found"
        }
    
    def _search_memory(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Optional[MemoryMatch]:
        """
        Search WorkingMemory for similar query.
        
        Args:
            query: Query to search for
            context: Optional context to include in embedding
            
        Returns:
            MemoryMatch if found, None otherwise
        """
        # Get all completed slots
        completed_slots = self.working_memory.get_all_completed()
        
        if not completed_slots:
            return None
        
        # Build query embedding
        search_text = query
        if context:
            search_text = f"{context}\n\n{query}"
        
        query_vec = self._get_query_embedding(search_text)
        
        # Search all slots for best match
        best_match = None
        best_score = 0.0
        
        for slot_id in completed_slots:
            score = self._compute_slot_similarity(slot_id, query_vec, query)
            
            if score > best_score:
                best_score = score
                result = self.working_memory.get(slot_id)
                if result:
                    best_match = MemoryMatch(
                        slot_id=slot_id,
                        expert_id=result.expert_id,
                        similarity=score,
                        content=result.raw_content,
                        is_cached=True
                    )
        
        return best_match if best_score > 0.5 else None  # Minimum threshold
    
    def _get_query_embedding(self, text: str) -> np.ndarray:
        """Get or compute query embedding with caching."""
        cache_key = hash(text) % 10000
        
        if cache_key not in self._query_cache:
            self._query_cache[cache_key] = self.embedding_manager.encode(
                text, normalize=True
            )
        
        return self._query_cache[cache_key]
    
    def _compute_slot_similarity(
        self,
        slot_id: str,
        query_vec: np.ndarray,
        query_text: str
    ) -> float:
        """
        Compute similarity between query and slot content.
        
        Uses multiple signals:
        1. Vector similarity of full content
        2. Keyword overlap in outline/sections
        3. Exact phrase matching
        """
        result = self.working_memory.get(slot_id)
        if not result:
            return 0.0
        
        # Skip very short content
        if len(result.raw_content) < self.min_content_length:
            return 0.0
        
        # Get or compute slot embedding
        if slot_id not in self._slot_embeddings:
            self._slot_embeddings[slot_id] = self.embedding_manager.encode(
                result.raw_content[:2000],  # First 2000 chars for efficiency
                normalize=True
            )
        
        slot_vec = self._slot_embeddings[slot_id]
        
        # Primary signal: cosine similarity
        vector_sim = float(np.dot(query_vec, slot_vec))
        
        # Secondary signal: keyword overlap in section titles
        query_words = set(query_text.lower().split())
        title_words = set()
        for title in result.section_titles.values():
            title_words.update(title.lower().split())
        
        if title_words:
            overlap = len(query_words & title_words) / max(len(query_words), 1)
            # Boost score if keywords match section titles
            vector_sim = vector_sim * 0.7 + overlap * 0.3
        
        return vector_sim
    
    def index_slot(self, slot_id: str, content: str):
        """
        Explicitly index a slot for memory search.
        Called when new slot output is stored.
        """
        self._slot_embeddings[slot_id] = self.embedding_manager.encode(
            content[:2000],
            normalize=True
        )
    
    def get_stats(self) -> Dict:
        """Get routing statistics."""
        total = self.stats["total_queries"]
        hit_rate = self.stats["memory_hits"] / total if total > 0 else 0
        
        return {
            **self.stats,
            "memory_hit_rate": hit_rate,
            "cached_embeddings": len(self._slot_embeddings)
        }
    
    def clear_cache(self):
        """Clear embedding caches."""
        self._query_cache.clear()
        self._slot_embeddings.clear()


def create_memory_router(
    working_memory: WorkingMemory,
    vector_router: VectorRouter,
    embedding_manager,
    config: Optional[Dict] = None
) -> MemoryVectorRouter:
    """
    Factory function to create MemoryVectorRouter with configuration.
    
    Args:
        working_memory: WorkingMemory instance
        vector_router: VectorRouter instance
        embedding_manager: EmbeddingManager instance
        config: Optional config dict with similarity_threshold, etc.
        
    Returns:
        Configured MemoryVectorRouter instance
    """
    config = config or {}
    
    return MemoryVectorRouter(
        working_memory=working_memory,
        vector_router=vector_router,
        embedding_manager=embedding_manager,
        similarity_threshold=config.get("similarity_threshold", 0.85),
        min_content_length=config.get("min_content_length", 100)
    )


if __name__ == "__main__":
    # Test memory router
    print("Testing MemoryVectorRouter...")
    
    from .working_memory import WorkingMemory
    from .embedding_manager import EmbeddingManager
    from .vector_router import VectorRouter
    
    # Initialize components
    em = EmbeddingManager(device="cpu")
    wm = WorkingMemory()
    vr = VectorRouter(embedding_manager=em)
    
    # Create memory router
    router = MemoryVectorRouter(
        working_memory=wm,
        vector_router=vr,
        embedding_manager=em,
        similarity_threshold=0.80
    )
    
    # Store some sample results
    sample_output_1 = """## Outline
1. Security Requirements
2. Implementation

---

## 1. Security Requirements
Use OAuth 2.0 with JWT tokens.

## 2. Implementation
Implement in Python with FastAPI.
"""
    
    sample_output_2 = """## Outline
1. Database Schema
2. Query Optimization

---

## 1. Database Schema
Use PostgreSQL with normalized tables.

## 2. Query Optimization
Add indexes on foreign keys.
"""
    
    wm.store("auth_design", "security_expert", sample_output_1)
    wm.store("db_design", "database_expert", sample_output_2)
    
    # Index slots
    router.index_slot("auth_design", sample_output_1)
    router.index_slot("db_design", sample_output_2)
    
    # Test routing
    test_queries = [
        "How should I implement authentication?",  # Should match auth_design
        "What database should I use?",  # Should match db_design
        "How to optimize React components?",  # No match, route to expert
    ]
    
    print("\nRouting test queries:")
    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: '{query}'")
        print(f"  Source: {result['source']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        if result['source'] == 'memory':
            print(f"  Matched slot: {result['match_info']['slot_id']}")
        else:
            print(f"  Routed to expert: {result['expert_id']}")
    
    print("\nStats:", router.get_stats())
