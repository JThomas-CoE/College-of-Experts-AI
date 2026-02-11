"""
V12 Vector Router - FAISS-based Expert and Template Routing
College of Experts Architecture

Handles:
- Expert routing via dual-embedding scoring (capability - λ×exclusion)
- Framework template lookup via FAISS index
- Confidence scoring and ambiguity detection
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[Warning] faiss not installed. Using numpy brute-force search.")

from .embedding_manager import EmbeddingManager, EmbeddingStore


class VectorRouter:
    """
    FAISS-based router for expert selection and template lookup.
    Uses dual-embedding scoring for experts: capability_score - λ × exclusion_score
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        expert_embeddings_path: str = "config/expert_embeddings",
        template_embeddings_path: str = "config/template_embeddings",
        expert_scopes_path: str = "config/expert_scopes"
    ):
        """
        Initialize vector router.
        
        Args:
            embedding_manager: Shared EmbeddingManager instance
            expert_embeddings_path: Path to expert embedding storage
            template_embeddings_path: Path to template embedding storage
            expert_scopes_path: Path to expert scope documents
        """
        self.embedding_manager = embedding_manager
        self.expert_path = Path(expert_embeddings_path)
        self.template_path = Path(template_embeddings_path)
        self.scopes_path = Path(expert_scopes_path)
        
        # Load expert embeddings
        self._load_expert_embeddings()
        
        # Load template index (if exists)
        self._load_template_index()
    
    def _load_expert_embeddings(self):
        """Load pre-computed expert embeddings."""
        cap_path = self.expert_path / "capability_vectors.npy"
        exc_path = self.expert_path / "exclusion_vectors.npy"
        ids_path = self.expert_path / "expert_ids.json"
        
        if cap_path.exists() and exc_path.exists() and ids_path.exists():
            self.expert_capability_vecs = np.load(cap_path)
            self.expert_exclusion_vecs = np.load(exc_path)
            with open(ids_path, 'r') as f:
                self.expert_ids = json.load(f)
            print(f"[VectorRouter] Loaded {len(self.expert_ids)} expert embeddings")
        else:
            print(f"[VectorRouter] Expert embeddings not found at {self.expert_path}")
            self.expert_capability_vecs = None
            self.expert_exclusion_vecs = None
            self.expert_ids = []
    
    def _load_template_index(self):
        """Load FAISS index for templates."""
        index_path = self.template_path / "template.index"
        ids_path = self.template_path / "template_ids.json"
        
        self.template_index = None
        self.template_ids = []
        
        if not index_path.exists():
            print(f"[VectorRouter] Template index not found at {index_path}")
            return
        
        if HAS_FAISS:
            self.template_index = faiss.read_index(str(index_path))
            with open(ids_path, 'r') as f:
                self.template_ids = json.load(f)
            print(f"[VectorRouter] Loaded FAISS index with {len(self.template_ids)} templates")
        else:
            # Fallback to numpy vectors
            vecs_path = self.template_path / "template_vectors.npy"
            if vecs_path.exists():
                self.template_vecs_fallback = np.load(vecs_path)
                with open(ids_path, 'r') as f:
                    self.template_ids = json.load(f)
                print(f"[VectorRouter] Loaded {len(self.template_ids)} templates (numpy fallback)")
    
    def route_to_expert(
        self,
        chunk_text: str,
        lambda_penalty: float = 0.3,
        return_scores: bool = False
    ) -> Tuple[str, float]:
        """
        Route a task chunk to the best matching expert.
        
        Uses dual-embedding scoring:
        score = cosine_sim(chunk, capability) - λ × cosine_sim(chunk, exclusion)
        
        Args:
            chunk_text: Task description to route
            lambda_penalty: Weight for exclusion penalty (0-1)
            return_scores: If True, return all expert scores
            
        Returns:
            Tuple of (expert_id, confidence_score) or (expert_id, confidence, all_scores)
        """
        if self.expert_capability_vecs is None:
            raise RuntimeError("Expert embeddings not loaded")
        
        # Encode chunk
        chunk_vec = self.embedding_manager.encode(chunk_text, normalize=True)
        
        # Compute capability scores (higher = better match)
        cap_scores = self.expert_capability_vecs @ chunk_vec
        
        # Compute exclusion scores (higher = worse match)
        exc_scores = self.expert_exclusion_vecs @ chunk_vec
        
        # Net score with exclusion penalty
        net_scores = cap_scores - lambda_penalty * exc_scores
        
        best_idx = np.argmax(net_scores)
        best_expert = self.expert_ids[best_idx]
        best_score = float(net_scores[best_idx])
        
        if return_scores:
            all_scores = {
                self.expert_ids[i]: {
                    "net": float(net_scores[i]),
                    "capability": float(cap_scores[i]),
                    "exclusion": float(exc_scores[i])
                }
                for i in range(len(self.expert_ids))
            }
            return best_expert, best_score, all_scores
        
        return best_expert, best_score
    
    def route_with_ambiguity_check(
        self,
        chunk_text: str,
        lambda_penalty: float = 0.3,
        ambiguity_threshold: float = 0.15,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Route chunk with ambiguity and confidence detection.
        
        Args:
            chunk_text: Task description to route
            lambda_penalty: Weight for exclusion penalty
            ambiguity_threshold: Score difference below which top-2 are "ambiguous"
            confidence_threshold: Minimum score for confident routing
            
        Returns:
            Dict with routing decision and metadata
        """
        expert_id, score, all_scores = self.route_to_expert(
            chunk_text, lambda_penalty, return_scores=True
        )
        
        # Sort experts by net score
        sorted_experts = sorted(
            all_scores.items(),
            key=lambda x: x[1]["net"],
            reverse=True
        )
        
        top1_id, top1_scores = sorted_experts[0]
        top2_id, top2_scores = sorted_experts[1] if len(sorted_experts) > 1 else (None, {"net": 0})
        
        score_gap = top1_scores["net"] - top2_scores["net"]
        
        result = {
            "expert_id": expert_id,
            "confidence": score,
            "is_ambiguous": score_gap < ambiguity_threshold,
            "is_low_confidence": score < confidence_threshold,
            "top_candidates": [
                {"id": top1_id, "score": top1_scores["net"]},
                {"id": top2_id, "score": top2_scores["net"]} if top2_id else None
            ],
            "all_scores": all_scores
        }
        
        if result["is_ambiguous"]:
            result["ambiguity_reason"] = f"Top-2 experts within {ambiguity_threshold}: {top1_id} ({top1_scores['net']:.3f}) vs {top2_id} ({top2_scores['net']:.3f})"
        
        if result["is_low_confidence"]:
            result["confidence_reason"] = f"Best score {score:.3f} below threshold {confidence_threshold}"
        
        return result
    
    def find_templates(
        self,
        query_text: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find top-k matching framework templates.
        
        Args:
            query_text: User query to match
            k: Number of templates to return
            
        Returns:
            List of (template_id, similarity_score) tuples
        """
        query_vec = self.embedding_manager.encode(query_text, normalize=True)
        
        if HAS_FAISS and self.template_index is not None:
            # FAISS search (L2 distance for normalized vectors = 2 - 2*cosine_sim)
            query_vec_2d = query_vec.reshape(1, -1)
            distances, indices = self.template_index.search(query_vec_2d, k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.template_ids):
                    # Convert L2 distance to cosine similarity
                    sim = 1 - dist / 2
                    results.append((self.template_ids[idx], float(sim)))
            
            return results
        
        elif hasattr(self, 'template_vecs_fallback'):
            # Numpy brute-force fallback
            similarities = self.template_vecs_fallback @ query_vec
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            
            return [
                (self.template_ids[i], float(similarities[i]))
                for i in top_k_idx
            ]
        
        else:
            print("[VectorRouter] No template index available")
            return []
    
    def get_expert_scope(self, expert_id: str) -> Optional[Dict]:
        """
        Load expert scope document for an expert.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            Dict with scope document contents or None
        """
        import yaml
        
        index_file = self.scopes_path / "index.json"
        if not index_file.exists():
            return None
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        for expert in index["experts"]:
            if expert["id"] == expert_id:
                scope_file = self.scopes_path / expert["scope_file"]
                if scope_file.exists():
                    with open(scope_file, 'r') as f:
                        return yaml.safe_load(f)
        
        return None


def build_faiss_index(
    vectors: np.ndarray,
    index_path: str,
    use_ivf: bool = False,
    nlist: int = 100
):
    """
    Build and save a FAISS index.
    
    Args:
        vectors: Embedding vectors (n, dim)
        index_path: Path to save index
        use_ivf: Use IVF index for large datasets
        nlist: Number of clusters for IVF
    """
    if not HAS_FAISS:
        print("[Warning] FAISS not available, skipping index build")
        return
    
    n, dim = vectors.shape
    
    if use_ivf and n > 1000:
        # IVF index for large datasets
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, n // 10))
        index.train(vectors)
        index.add(vectors)
    else:
        # Flat index for small datasets
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
    
    faiss.write_index(index, index_path)
    print(f"[FAISS] Saved index with {n} vectors to {index_path}")


if __name__ == "__main__":
    # Test vector router
    print("Testing VectorRouter...")
    
    # Initialize embedding manager
    from embedding_manager import EmbeddingManager
    em = EmbeddingManager(device="cuda")
    
    # Initialize router
    router = VectorRouter(
        embedding_manager=em,
        expert_embeddings_path="config/expert_embeddings",
        template_embeddings_path="config/template_embeddings"
    )
    
    # Test routing
    test_chunks = [
        "Write a Python function to encrypt sensitive data",
        "Draft a Delaware jurisdiction liability disclaimer",
        "Design a secure authentication system for healthcare records",
        "Explain the symptoms and treatment of type 2 diabetes",
        "Write a SQL query to join patient and appointment tables",
        "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 1"
    ]
    
    print("\nRouting test chunks:")
    for chunk in test_chunks:
        result = router.route_with_ambiguity_check(chunk)
        status = ""
        if result["is_ambiguous"]:
            status = " [AMBIGUOUS]"
        if result["is_low_confidence"]:
            status += " [LOW CONF]"
        print(f"  '{chunk[:50]}...' -> {result['expert_id']} ({result['confidence']:.3f}){status}")
