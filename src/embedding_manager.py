"""
V12 Embedding Manager - BGE-M3 Loading and Embedding Generation
College of Experts Architecture

Handles:
- BGE-M3 model loading (resident for session)
- Text embedding generation
- Pre-computed embedding storage and retrieval
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[Warning] sentence-transformers not installed. Embedding features disabled.")


class EmbeddingManager:
    """
    Manages BGE-M3 embedding model and embedding operations.
    Model is loaded once at startup and kept resident.
    """
    
    DEFAULT_MODEL = "BAAI/bge-m3"
    EMBED_DIM = 1024  # BGE-M3 dimension
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        """
        Initialize embedding manager with specified model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model into memory."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("[EmbeddingManager] sentence-transformers not available, using mock mode")
            return
        
        print(f"[EmbeddingManager] Loading {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[EmbeddingManager] Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[EmbeddingManager] Failed to load on {self.device}, trying CPU: {e}")
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device="cpu")
            print("[EmbeddingManager] Model loaded on CPU")
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into embedding vector(s).
        
        Args:
            texts: Single string or list of strings to encode
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
            
        Returns:
            np.ndarray: Embedding vector(s) of shape (embed_dim,) or (n, embed_dim)
        """
        if self.model is None:
            # Mock mode - return random vectors for testing
            if isinstance(texts, str):
                return np.random.randn(self.EMBED_DIM).astype(np.float32)
            return np.random.randn(len(texts), self.EMBED_DIM).astype(np.float32)
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings.astype(np.float32)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode a large batch of texts efficiently.
        
        Args:
            texts: List of strings to encode
            batch_size: Number of texts per batch
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            np.ndarray: Embedding vectors of shape (n, embed_dim)
        """
        if self.model is None:
            return np.random.randn(len(texts), self.EMBED_DIM).astype(np.float32)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings.astype(np.float32)


class EmbeddingStore:
    """
    Manages pre-computed embedding storage and retrieval.
    Handles versioning and cache invalidation.
    """
    
    def __init__(self, store_path: str):
        """
        Initialize embedding store.
        
        Args:
            store_path: Directory path for embedding storage
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.store_path / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load or initialize metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "embedding_model": None,
            "generated_at": None,
            "source_hash": None,
            "embed_dim": None,
            "count": 0
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_source_hash(self, source_files: List[str]) -> str:
        """Compute hash of source files for cache invalidation."""
        hasher = hashlib.md5()
        for filepath in sorted(source_files):
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()[:16]
    
    def needs_rebuild(self, source_files: List[str], model_name: str) -> bool:
        """
        Check if embeddings need to be rebuilt.
        
        Args:
            source_files: List of source file paths
            model_name: Embedding model name
            
        Returns:
            bool: True if rebuild needed
        """
        if not (self.store_path / "vectors.npy").exists():
            return True
        
        if self.metadata.get("embedding_model") != model_name:
            return True
        
        current_hash = self._compute_source_hash(source_files)
        if self.metadata.get("source_hash") != current_hash:
            return True
        
        return False
    
    def save_embeddings(
        self,
        vectors: np.ndarray,
        ids: List[str],
        model_name: str,
        source_files: List[str],
        vector_name: str = "vectors"
    ):
        """
        Save embeddings and metadata.
        
        Args:
            vectors: Embedding vectors (n, embed_dim)
            ids: List of IDs corresponding to vectors
            model_name: Embedding model used
            source_files: Source files used to generate embeddings
            vector_name: Name for the vector file (default: "vectors")
        """
        # Save vectors
        np.save(self.store_path / f"{vector_name}.npy", vectors)
        
        # Save IDs
        with open(self.store_path / f"{vector_name}_ids.json", 'w') as f:
            json.dump(ids, f)
        
        # Update metadata
        self.metadata = {
            "embedding_model": model_name,
            "generated_at": datetime.now().isoformat(),
            "source_hash": self._compute_source_hash(source_files),
            "embed_dim": vectors.shape[1] if len(vectors.shape) > 1 else len(vectors),
            "count": len(ids)
        }
        self._save_metadata()
        
        print(f"[EmbeddingStore] Saved {len(ids)} embeddings to {self.store_path}")
    
    def load_embeddings(self, vector_name: str = "vectors") -> tuple:
        """
        Load embeddings and IDs.
        
        Args:
            vector_name: Name of the vector file to load
            
        Returns:
            tuple: (vectors np.ndarray, ids List[str])
        """
        vectors_path = self.store_path / f"{vector_name}.npy"
        ids_path = self.store_path / f"{vector_name}_ids.json"
        
        if not vectors_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Embeddings not found at {self.store_path}")
        
        vectors = np.load(vectors_path)
        with open(ids_path, 'r') as f:
            ids = json.load(f)
        
        return vectors, ids


def build_expert_embeddings(
    scope_dir: str,
    output_dir: str,
    embedding_manager: EmbeddingManager
):
    """
    Build embeddings for expert scope documents.
    
    Args:
        scope_dir: Directory containing expert scope YAML files
        output_dir: Directory to save embeddings
        embedding_manager: EmbeddingManager instance
    """
    import yaml
    
    scope_path = Path(scope_dir)
    index_file = scope_path / "index.json"
    
    if not index_file.exists():
        raise FileNotFoundError(f"Expert index not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    expert_ids = []
    capability_texts = []
    exclusion_texts = []
    source_files = [str(index_file)]
    
    for expert in index["experts"]:
        scope_file = scope_path / expert["scope_file"]
        source_files.append(str(scope_file))
        
        with open(scope_file, 'r') as f:
            scope = yaml.safe_load(f)
        
        expert_ids.append(scope["expert_id"])
        capability_texts.append(scope["capability_scope"])
        exclusion_texts.append(scope["exclusion_scope"])
    
    print(f"[BuildEmbeddings] Encoding {len(expert_ids)} expert scopes...")
    
    # Encode capability scopes
    capability_vecs = embedding_manager.encode_batch(capability_texts)
    
    # Encode exclusion scopes
    exclusion_vecs = embedding_manager.encode_batch(exclusion_texts)
    
    # Save to store
    store = EmbeddingStore(output_dir)
    
    # Save capability vectors
    store.save_embeddings(
        capability_vecs,
        expert_ids,
        embedding_manager.model_name,
        source_files,
        vector_name="capability_vectors"
    )
    
    # Save exclusion vectors separately
    np.save(Path(output_dir) / "exclusion_vectors.npy", exclusion_vecs)
    
    # Save expert IDs
    with open(Path(output_dir) / "expert_ids.json", 'w') as f:
        json.dump(expert_ids, f)
    
    print(f"[BuildEmbeddings] Expert embeddings saved to {output_dir}")
    
    return capability_vecs, exclusion_vecs, expert_ids


if __name__ == "__main__":
    # Test embedding manager
    print("Testing EmbeddingManager...")
    
    em = EmbeddingManager(device="cuda")
    
    # Test single encoding
    vec = em.encode("This is a test sentence.")
    print(f"Single encoding shape: {vec.shape}")
    
    # Test batch encoding
    texts = [
        "Write a Python function to sort a list",
        "Draft a legal disclaimer for software",
        "Explain the symptoms of diabetes"
    ]
    vecs = em.encode(texts)
    print(f"Batch encoding shape: {vecs.shape}")
    
    # Test similarity
    from numpy.linalg import norm
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    print(f"\nSimilarity scores:")
    print(f"  Python-Legal: {cosine_sim(vecs[0], vecs[1]):.3f}")
    print(f"  Python-Medical: {cosine_sim(vecs[0], vecs[2]):.3f}")
    print(f"  Legal-Medical: {cosine_sim(vecs[1], vecs[2]):.3f}")
