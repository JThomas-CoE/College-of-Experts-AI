"""
Vector Backbone - Grounded knowledge layer using ChromaDB.
Provides long-term semantic retrieval and grounded context for experts.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

class VectorBackbone:
    """
    Manages local vector storage for grounded knowledge (RAG).
    Enables experts to cite sources and avoid hallucinations.
    """
    
    def __init__(self, db_path: str = "data/vector_db"):
        self.db_path = db_path
        if not CHROMA_AVAILABLE:
            print("[VectorBackbone] Warning: chromadb and dependencies not found. RAG disabled.")
            self.client = None
            self.collections = {}
            return

        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collections: Dict[str, Any] = {}

    def get_or_create_collection(self, name: str) -> Any:
        if not CHROMA_AVAILABLE or self.client is None:
            return None
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_func
            )
        return self.collections[name]

    def ingest_documents(self, collection_name: str, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        collection = self.get_or_create_collection(collection_name)
        if collection:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, collection_name: str, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        collection = self.get_or_create_collection(collection_name)
        if collection:
            return collection.query(query_texts=[query_text], n_results=n_results)
        return {"documents": [[]]}

    def delete_collection(self, name: str):
        if CHROMA_AVAILABLE and self.client:
            self.client.delete_collection(name=name)
            if name in self.collections:
                del self.collections[name]

    def get_stats(self) -> Dict[str, int]:
        if not CHROMA_AVAILABLE or self.client is None:
            return {}
        stats = {}
        for name in self.client.list_collections():
            try:
                col = self.client.get_collection(name.name)
                stats[name.name] = col.count()
            except: pass
        return stats

if __name__ == "__main__":
    # Integration smoke test
    vb = VectorBackbone("data/test_vector_db")
    if CHROMA_AVAILABLE:
        # Test Ingestion
        vb.ingest_documents(
            "legal",
            ["Section 230 provides immunity for online intermediaries."],
            [{"source": "US Code", "section": "230"}],
            ["id1"]
        )
        
        # Test Query
        results = vb.query("legal", "What is Section 230?")
        print(f"Results: {results['documents']}")
        
    print(f"Stats: {vb.get_stats()}")
