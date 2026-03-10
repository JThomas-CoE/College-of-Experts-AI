"""
Three-Tier Knowledge Layer for College of Experts

Tier 1: Working Memory (session-scoped)
    - Read-only access to finished subtask results
    - Integration context from other slots
    - Ephemeral, per-session

Tier 2: Local Knowledge Base (disk-backed, RAM-indexed)
    - Curated encyclopedia of established facts
    - Coding patterns, best practices, templates
    - Few GB max, periodically updated
    - FAISS index in RAM for fast semantic search

Tier 3: Web Search (real-time)
    - Current/dynamic knowledge
    - Web search APIs (Brave, SerpAPI)
    - Web-accessible knowledge bases
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KnowledgeChunk:
    """A single piece of knowledge."""
    id: str
    content: str
    category: str  # coding, ui_ux, security, database, general, etc.
    subcategory: str  # python, javascript, css, sql, etc.
    tags: List[str]
    source: str  # "local", "web", "memory"
    source_url: Optional[str] = None
    created_at: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "source": self.source,
            "source_url": self.source_url,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeChunk":
        return cls(
            id=d["id"],
            content=d["content"],
            category=d.get("category", "general"),
            subcategory=d.get("subcategory", ""),
            tags=d.get("tags", []),
            source=d.get("source", "local"),
            source_url=d.get("source_url"),
            created_at=d.get("created_at")
        )


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""
    chunks: List[KnowledgeChunk]
    scores: List[float]
    tier: str  # "memory", "local", "web"
    query: str
    retrieval_time_ms: float


# =============================================================================
# TIER 1: WORKING MEMORY
# =============================================================================

class WorkingMemoryRetriever:
    """
    Tier 1: Retrieve context from session working memory.
    
    Provides read-only access to finished subtask results for integration.
    """
    
    def __init__(self, working_memory: Any):
        """
        Args:
            working_memory: WorkingMemory instance from the session
        """
        self.memory = working_memory
    
    def get_completed_slots(self) -> Dict[str, str]:
        """Get all completed slot results."""
        results = {}
        if self.memory and hasattr(self.memory, 'results'):
            for slot_id, result in self.memory.results.items():
                if result and hasattr(result, 'raw_content') and result.raw_content:
                    results[slot_id] = result.raw_content
        return results
    
    def get_slot_result(self, slot_id: str) -> Optional[str]:
        """Get a specific slot's result."""
        if self.memory and hasattr(self.memory, 'get'):
            result = self.memory.get(slot_id)
            if result and hasattr(result, 'raw_content'):
                return result.raw_content
        return None
    
    def search_memory(self, query: str, embedding_fn: callable, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Semantic search through completed slot results.
        
        Returns: List of (slot_id, content, similarity_score)
        """
        completed = self.get_completed_slots()
        if not completed:
            return []
        
        query_vec = embedding_fn(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        results = []
        for slot_id, content in completed.items():
            # Embed content (or use cached embedding)
            content_vec = embedding_fn(content[:2000])  # Truncate for embedding
            content_vec = content_vec / (np.linalg.norm(content_vec) + 1e-8)
            
            similarity = float(np.dot(query_vec, content_vec))
            results.append((slot_id, content, similarity))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# =============================================================================
# TIER 2: LOCAL KNOWLEDGE BASE
# =============================================================================

class LocalKnowledgeBase:
    """
    Tier 2: Disk-backed knowledge base with RAM-indexed search.
    
    Structure:
        data/knowledge/
            index.json          # Metadata and chunk registry
            embeddings.npy      # Precomputed embeddings
            chunks/
                coding_python.json
                coding_javascript.json
                ui_modern_dark.json
                security_auth.json
                database_sql.json
                ...
    """
    
    def __init__(
        self, 
        data_dir: str = "data/knowledge",
        embedding_fn: Optional[callable] = None,
        load_to_ram: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.embedding_fn = embedding_fn
        self.load_to_ram = load_to_ram
        
        # In-memory storage
        self.chunks: Dict[str, KnowledgeChunk] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        self.faiss_index = None
        
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "chunks").mkdir(exist_ok=True)
        
        # Load if exists
        if (self.data_dir / "index.json").exists():
            self._load()
    
    def _load(self):
        """Load knowledge base into memory."""
        import time
        start = time.time()
        
        # Load index
        with open(self.data_dir / "index.json") as f:
            index = json.load(f)
        
        self.chunk_ids = index.get("chunk_ids", [])
        
        # Load chunks
        for chunk_file in (self.data_dir / "chunks").glob("*.json"):
            with open(chunk_file) as f:
                data = json.load(f)
                for chunk_data in data.get("chunks", []):
                    chunk = KnowledgeChunk.from_dict(chunk_data)
                    self.chunks[chunk.id] = chunk
        
        # Load embeddings
        emb_path = self.data_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            self._build_faiss_index()
        
        elapsed = (time.time() - start) * 1000
        logger.info(f"[KnowledgeBase] Loaded {len(self.chunks)} chunks in {elapsed:.1f}ms")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search.
        
        Uses CPU FAISS (not faiss-gpu) to keep index in system RAM.
        This preserves VRAM for model inference.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        
        try:
            import faiss
            dim = self.embeddings.shape[1]
            # CPU index - system RAM only
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
            
            # Normalize embeddings (all operations in CPU/numpy)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            normalized = self.embeddings / (norms + 1e-8)
            
            self.faiss_index.add(normalized.astype(np.float32))
            logger.info(f"[KnowledgeBase] Built CPU FAISS index with {self.faiss_index.ntotal} vectors")
        except ImportError:
            logger.warning("[KnowledgeBase] FAISS not available, using numpy fallback")
            self.faiss_index = None
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Semantic search through knowledge base.
        
        All operations use CPU/system RAM to preserve VRAM for model inference.
        
        Args:
            query: Search query
            top_k: Number of results
            category: Filter by category
            tags: Filter by tags (any match)
        
        Returns:
            List of (chunk, similarity_score)
        """
        if not self.chunks or self.embedding_fn is None:
            return []
        
        # Embed query and ensure CPU numpy array
        query_vec = self.embedding_fn(query)
        if hasattr(query_vec, 'cpu'):  # PyTorch tensor
            query_vec = query_vec.cpu().numpy()
        elif hasattr(query_vec, 'get'):  # CuPy array
            query_vec = query_vec.get()
        query_vec = np.asarray(query_vec, dtype=np.float32).flatten()
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        if self.faiss_index is not None and self.embeddings is not None:
            # Fast FAISS search
            scores, indices = self.faiss_index.search(
                query_vec.reshape(1, -1).astype(np.float32), 
                min(top_k * 3, len(self.chunk_ids))  # Over-fetch for filtering
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.chunk_ids):
                    continue
                chunk_id = self.chunk_ids[idx]
                chunk = self.chunks.get(chunk_id)
                if chunk is None:
                    continue
                
                # Apply filters
                if category and chunk.category != category:
                    continue
                if tags and not any(t in chunk.tags for t in tags):
                    continue
                
                results.append((chunk, float(score)))
                if len(results) >= top_k:
                    break
            
            return results
        else:
            # Numpy fallback
            results = []
            for chunk_id, chunk in self.chunks.items():
                # Apply filters
                if category and chunk.category != category:
                    continue
                if tags and not any(t in chunk.tags for t in tags):
                    continue
                
                # Get embedding
                idx = self.chunk_ids.index(chunk_id) if chunk_id in self.chunk_ids else -1
                if idx >= 0 and self.embeddings is not None:
                    chunk_vec = self.embeddings[idx]
                    chunk_vec = chunk_vec / (np.linalg.norm(chunk_vec) + 1e-8)
                    score = float(np.dot(query_vec, chunk_vec))
                    results.append((chunk, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def add_chunk(self, chunk: KnowledgeChunk):
        """Add a knowledge chunk."""
        self.chunks[chunk.id] = chunk
        if chunk.id not in self.chunk_ids:
            self.chunk_ids.append(chunk.id)
    
    def save(self):
        """Save knowledge base to disk."""
        # Group chunks by category
        by_category: Dict[str, List[dict]] = {}
        for chunk in self.chunks.values():
            key = f"{chunk.category}_{chunk.subcategory}" if chunk.subcategory else chunk.category
            if key not in by_category:
                by_category[key] = []
            by_category[key].append(chunk.to_dict())
        
        # Save chunk files
        for key, chunks in by_category.items():
            filename = key.replace("/", "_").replace(" ", "_").lower()
            with open(self.data_dir / "chunks" / f"{filename}.json", "w") as f:
                json.dump({"chunks": chunks}, f, indent=2)
        
        # Compute and save embeddings (CPU/system RAM only to preserve VRAM)
        if self.embedding_fn and self.chunks:
            embeddings = []
            for chunk_id in self.chunk_ids:
                chunk = self.chunks.get(chunk_id)
                if chunk:
                    vec = self.embedding_fn(chunk.content[:2000])
                    # Ensure CPU numpy array
                    if hasattr(vec, 'cpu'):  # PyTorch tensor
                        vec = vec.cpu().numpy()
                    elif hasattr(vec, 'get'):  # CuPy array
                        vec = vec.get()
                    vec = np.asarray(vec, dtype=np.float32).flatten()
                    embeddings.append(vec)
            
            self.embeddings = np.array(embeddings, dtype=np.float32)
            np.save(self.data_dir / "embeddings.npy", self.embeddings)
            self._build_faiss_index()
        
        # Save index
        index = {
            "chunk_ids": self.chunk_ids,
            "total_chunks": len(self.chunks),
            "updated_at": datetime.now().isoformat()
        }
        with open(self.data_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"[KnowledgeBase] Saved {len(self.chunks)} chunks")
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        categories = {}
        for chunk in self.chunks.values():
            cat = chunk.category
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "categories": categories,
            "has_embeddings": self.embeddings is not None,
            "faiss_indexed": self.faiss_index is not None
        }


# =============================================================================
# TIER 3: WEB SEARCH
# =============================================================================

class WebSearchProvider:
    """
    Tier 3: Web search for current knowledge.
    
    Supports multiple backends:
    - Brave Search API
    - DuckDuckGo (no API key)
    - Wikipedia API (no API key)
    """
    
    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        enable_duckduckgo: bool = True,
        enable_wikipedia: bool = True
    ):
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.enable_duckduckgo = enable_duckduckgo
        self.enable_wikipedia = enable_wikipedia
        
        # Check available backends
        self.available_backends = []
        if self.brave_api_key:
            self.available_backends.append("brave")
        if enable_duckduckgo:
            self.available_backends.append("duckduckgo")
        if enable_wikipedia:
            self.available_backends.append("wikipedia")
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        backend: Optional[str] = None
    ) -> List[KnowledgeChunk]:
        """
        Search the web for current knowledge.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            backend: Specific backend to use (or auto-select)
        
        Returns:
            List of KnowledgeChunk from web sources
        """
        if not self.available_backends:
            logger.warning("[WebSearch] No search backends available")
            return []
        
        backend = backend or self.available_backends[0]
        
        try:
            if backend == "brave" and self.brave_api_key:
                return await self._search_brave(query, max_results)
            elif backend == "duckduckgo":
                return await self._search_duckduckgo(query, max_results)
            elif backend == "wikipedia":
                return await self._search_wikipedia(query, max_results)
            else:
                logger.warning(f"[WebSearch] Unknown backend: {backend}")
                return []
        except Exception as e:
            logger.error(f"[WebSearch] Search failed: {e}")
            return []
    
    async def _search_brave(self, query: str, max_results: int) -> List[KnowledgeChunk]:
        """Search using Brave Search API."""
        import aiohttp
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
        params = {
            "q": query,
            "count": max_results
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"[WebSearch] Brave API error: {resp.status}")
                    return []
                
                data = await resp.json()
                results = []
                
                for item in data.get("web", {}).get("results", []):
                    chunk = KnowledgeChunk(
                        id=f"web_brave_{hashlib.md5(item['url'].encode()).hexdigest()[:12]}",
                        content=f"{item.get('title', '')}\n\n{item.get('description', '')}",
                        category="web",
                        subcategory="search",
                        tags=["web", "brave"],
                        source="web",
                        source_url=item.get("url"),
                        created_at=datetime.now().isoformat()
                    )
                    results.append(chunk)
                
                return results
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[KnowledgeChunk]:
        """Search using DuckDuckGo (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("[WebSearch] duckduckgo_search not installed")
            return []
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                chunk = KnowledgeChunk(
                    id=f"web_ddg_{hashlib.md5(r['href'].encode()).hexdigest()[:12]}",
                    content=f"{r.get('title', '')}\n\n{r.get('body', '')}",
                    category="web",
                    subcategory="search",
                    tags=["web", "duckduckgo"],
                    source="web",
                    source_url=r.get("href"),
                    created_at=datetime.now().isoformat()
                )
                results.append(chunk)
        
        return results
    
    async def _search_wikipedia(self, query: str, max_results: int) -> List[KnowledgeChunk]:
        """Search Wikipedia API."""
        import aiohttp
        
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                
                data = await resp.json()
                results = []
                
                for item in data.get("query", {}).get("search", []):
                    # Strip HTML tags from snippet
                    import re
                    snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                    
                    chunk = KnowledgeChunk(
                        id=f"web_wiki_{item['pageid']}",
                        content=f"{item.get('title', '')}\n\n{snippet}",
                        category="web",
                        subcategory="wikipedia",
                        tags=["web", "wikipedia", "encyclopedia"],
                        source="web",
                        source_url=f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                        created_at=datetime.now().isoformat()
                    )
                    results.append(chunk)
                
                return results


# =============================================================================
# UNIFIED KNOWLEDGE RETRIEVER
# =============================================================================

class KnowledgeRetriever:
    """
    Unified three-tier knowledge retrieval.
    
    Searches tiers in order:
    1. Working Memory (session context)
    2. Local Knowledge Base (curated facts)
    3. Web Search (current knowledge)
    
    Results are merged and ranked by relevance.
    """
    
    def __init__(
        self,
        embedding_fn: callable,
        knowledge_base_dir: str = "data/knowledge",
        brave_api_key: Optional[str] = None,
        enable_web_search: bool = True
    ):
        self.embedding_fn = embedding_fn
        
        # Initialize tiers
        self.local_kb = LocalKnowledgeBase(
            data_dir=knowledge_base_dir,
            embedding_fn=embedding_fn
        )
        
        self.web_search = WebSearchProvider(
            brave_api_key=brave_api_key,
            enable_duckduckgo=enable_web_search,
            enable_wikipedia=enable_web_search
        ) if enable_web_search else None
        
        # Working memory is set per-session
        self.memory_retriever: Optional[WorkingMemoryRetriever] = None
    
    def set_working_memory(self, memory: Any):
        """Set the working memory for the current session."""
        self.memory_retriever = WorkingMemoryRetriever(memory)
    
    async def retrieve(
        self,
        query: str,
        slot_context: Optional[dict] = None,
        top_k: int = 5,
        use_memory: bool = True,
        use_local: bool = True,
        use_web: bool = False,  # Disabled by default for speed
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: float = 0.0  # Minimum similarity score
    ) -> Dict[str, List[Tuple[KnowledgeChunk, float]]]:
        """
        Retrieve knowledge from all available tiers.
        
        Args:
            query: Search query
            slot_context: Optional context about current slot (title, description)
            top_k: Results per tier
            use_memory: Search working memory
            use_local: Search local knowledge base
            use_web: Search web (slower)
            category: Filter local KB by category
            tags: Filter local KB by tags
        
        Returns:
            Dict mapping tier name to list of (chunk, score)
        """
        import time
        results = {}
        
        # Enhance query with slot context
        enhanced_query = query
        if slot_context:
            enhanced_query = f"{slot_context.get('title', '')} {slot_context.get('description', '')} {query}"
        
        # Tier 1: Working Memory
        if use_memory and self.memory_retriever:
            start = time.time()
            memory_results = self.memory_retriever.search_memory(
                enhanced_query, 
                self.embedding_fn, 
                top_k
            )
            
            chunks = []
            for slot_id, content, score in memory_results:
                if score < min_score:
                    continue
                chunk = KnowledgeChunk(
                    id=f"memory_{slot_id}",
                    content=content,
                    category="memory",
                    subcategory="slot_result",
                    tags=["session", "slot", slot_id],
                    source="memory"
                )
                chunks.append((chunk, score))
            
            results["memory"] = chunks
            logger.debug(f"[Knowledge] Memory search: {len(chunks)} results in {(time.time()-start)*1000:.1f}ms")
        
        # Tier 2: Local Knowledge Base
        if use_local and self.local_kb.chunks:
            start = time.time()
            local_results = self.local_kb.search(
                enhanced_query,
                top_k=top_k,
                category=category,
                tags=tags
            )
            # Filter by score
            if min_score > 0:
                local_results = [(c, s) for c, s in local_results if s >= min_score]
                
            results["local"] = local_results
            logger.debug(f"[Knowledge] Local search: {len(local_results)} results in {(time.time()-start)*1000:.1f}ms")
        
        # Tier 3: Web Search
        if use_web and self.web_search:
            start = time.time()
            web_results = await self.web_search.search(query, max_results=top_k)
            # Web results don't have scores, assign based on position
            results["web"] = [(chunk, 1.0 - i * 0.1) for i, chunk in enumerate(web_results)]
            logger.debug(f"[Knowledge] Web search: {len(web_results)} results in {(time.time()-start)*1000:.1f}ms")
        
        return results
    
    def format_context(
        self, 
        results: Dict[str, List[Tuple[KnowledgeChunk, float]]],
        max_tokens: int = 2000
    ) -> str:
        """
        Format retrieved knowledge into context string for LLM prompt.
        
        Args:
            results: Results from retrieve()
            max_tokens: Approximate max length (chars * 0.3 ≈ tokens)
        
        Returns:
            Formatted context string
        """
        max_chars = int(max_tokens / 0.3)
        
        sections = []
        total_chars = 0
        
        # Priority order: memory > local > web
        for tier in ["memory", "local", "web"]:
            if tier not in results:
                continue
            
            tier_chunks = results[tier]
            if not tier_chunks:
                continue
            
            tier_label = {
                "memory": "From Previous Subtasks",
                "local": "Reference Knowledge",
                "web": "Web Search Results"
            }.get(tier, tier.title())
            
            section_parts = [f"\n### {tier_label}:\n"]
            
            for chunk, score in tier_chunks:
                content = chunk.content
                if total_chars + len(content) > max_chars:
                    # Truncate
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        content = content[:remaining] + "..."
                    else:
                        break
                
                section_parts.append(f"[{chunk.category}/{chunk.subcategory}] (relevance: {score:.2f})")
                section_parts.append(content)
                section_parts.append("")
                total_chars += len(content)
            
            if len(section_parts) > 1:
                sections.append("\n".join(section_parts))
        
        return "\n".join(sections) if sections else ""


# =============================================================================
# KNOWLEDGE BASE SEEDING
# =============================================================================

def create_seed_knowledge() -> List[KnowledgeChunk]:
    """Create initial seed knowledge for the knowledge base."""
    
    chunks = []
    
    # Modern Dark Theme UI Pattern
    chunks.append(KnowledgeChunk(
        id="ui_modern_dark_001",
        content='''Modern Dark Theme UI Best Practices (2024+):

COLOR PALETTE:
- Background: #0f0f0f (near-black) or #1a1a2e (dark blue-gray)
- Surface/Cards: #1e1e2e or #16213e  
- Primary accent: #6366f1 (indigo) or #8b5cf6 (violet)
- Success: #22c55e (green)
- Error: #ef4444 (red)
- Text primary: #f1f5f9 (off-white)
- Text secondary: #94a3b8 (muted)
- Border: #334155 (subtle)

CSS VARIABLES EXAMPLE:
```css
:root {
    --bg-primary: #0f0f0f;
    --bg-secondary: #1a1a2e;
    --bg-card: #1e1e2e;
    --accent-primary: #6366f1;
    --accent-hover: #818cf8;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    --radius: 0.75rem;
}
```

TYPOGRAPHY:
- Font: Inter, system-ui, -apple-system, sans-serif
- Headings: font-weight: 600-700
- Body: font-weight: 400, line-height: 1.6
- Use font-smoothing: antialiased

SPACING:
- Use 8px grid system (0.5rem increments)
- Card padding: 1.5rem (24px)
- Section gaps: 2rem (32px)

EFFECTS:
- Subtle gradients on buttons/cards
- Box shadows with low opacity
- Border-radius: 8-16px for modern feel
- Transitions: 150-200ms ease-out

ACCESSIBILITY:
- Ensure 4.5:1 contrast ratio minimum
- Focus states with visible outlines
- Reduce motion for prefers-reduced-motion''',
        category="ui_ux",
        subcategory="dark_theme",
        tags=["css", "dark-mode", "color-palette", "modern", "design-system"],
        source="local",
        created_at=datetime.now().isoformat()
    ))
    
    # Generic Dark Theme Component Patterns (applies to ANY web app)
    chunks.append(KnowledgeChunk(
        id="ui_component_patterns_001",
        content='''Modern Dark Theme Component Patterns (Generic - Apply to ANY Web App):

## HTML5 BOILERPLATE
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>App Title</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg: #0f0f0f;
            --surface: #1a1a2e;
            --card: #1e1e2e;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --success: #22c55e;
            --danger: #ef4444;
            --warning: #f59e0b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
            --radius: 0.75rem;
        }
        
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
    </style>
</head>
```

## HEADER WITH GRADIENT TITLE
```css
header { text-align: center; margin-bottom: 2rem; }

h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle { color: var(--text-muted); font-size: 1.1rem; }
```

## CARD COMPONENT
```css
.card {
    background: var(--card);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}
```

## FORM INPUTS (text, textarea, select, date)
```css
input, select, textarea {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    color: var(--text);
    font-size: 1rem;
    transition: border-color 0.2s;
}

input:focus, select:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
}

input::placeholder, textarea::placeholder { color: var(--text-muted); }

/* Date/datetime inputs */
input[type="date"], input[type="datetime-local"] {
    color-scheme: dark;
}

/* Form layout */
.form-group { margin-bottom: 1rem; }
.form-row { display: flex; gap: 1rem; flex-wrap: wrap; }
label { display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem; }
```

## BUTTONS
```css
button, .btn {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
}

button:hover { background: var(--accent-hover); }
button:active { transform: scale(0.98); }

.btn-danger { background: var(--danger); }
.btn-success { background: var(--success); }
.btn-outline {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
}
```

## LIST ITEMS
```css
.list-container { display: flex; flex-direction: column; gap: 0.75rem; }

.list-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    transition: background 0.2s;
}

.list-item:hover { background: var(--card); }
.list-item .content { flex: 1; }
.list-item .meta { font-size: 0.875rem; color: var(--text-muted); }
.list-item .actions { display: flex; gap: 0.5rem; }
```

## STATUS COLORS
```css
.status-success, .priority-low { color: var(--success); }
.status-warning, .priority-medium { color: var(--warning); }
.status-danger, .priority-high { color: var(--danger); }
```

## RESPONSIVE
```css
@media (max-width: 600px) {
    .container { padding: 1rem; }
    .form-row { flex-direction: column; }
    h1 { font-size: 2rem; }
}
```

IMPORTANT: Include ALL form fields that the user requests. If they ask for title, description, priority, and due_date fields, include ALL of them in the form.''',
        category="ui_ux",
        subcategory="components",
        tags=["css", "html", "components", "forms", "dark-theme", "generic"],
        source="local",
        created_at=datetime.now().isoformat()
    ))
    
    # Flask REST API Pattern
    chunks.append(KnowledgeChunk(
        id="coding_flask_rest_001",
        content='''Flask REST API Best Practices (2024):

```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from functools import wraps
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')

db = SQLAlchemy(app)

# ========== MODELS ==========

class Task(db.Model):
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    priority = db.Column(db.String(20), default='medium')
    due_date = db.Column(db.DateTime)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'completed': self.completed,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# ========== ERROR HANDLERS ==========

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error', 'message': 'Internal server error'}), 500

# ========== API ROUTES ==========

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """List all tasks with optional filtering."""
    completed = request.args.get('completed', type=lambda x: x.lower() == 'true')
    priority = request.args.get('priority')
    
    query = Task.query
    if completed is not None:
        query = query.filter_by(completed=completed)
    if priority:
        query = query.filter_by(priority=priority)
    
    tasks = query.order_by(Task.created_at.desc()).all()
    return jsonify([t.to_dict() for t in tasks])

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new task."""
    data = request.get_json()
    
    if not data or not data.get('title'):
        return jsonify({'error': 'Title is required'}), 400
    
    task = Task(
        title=data['title'],
        description=data.get('description'),
        priority=data.get('priority', 'medium'),
        due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None
    )
    
    db.session.add(task)
    db.session.commit()
    
    return jsonify(task.to_dict()), 201

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """Get a single task by ID."""
    task = Task.query.get_or_404(task_id)
    return jsonify(task.to_dict())

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    """Update an existing task."""
    task = Task.query.get_or_404(task_id)
    data = request.get_json()
    
    if 'title' in data:
        task.title = data['title']
    if 'description' in data:
        task.description = data['description']
    if 'priority' in data:
        task.priority = data['priority']
    if 'completed' in data:
        task.completed = data['completed']
    if 'due_date' in data:
        task.due_date = datetime.fromisoformat(data['due_date']) if data['due_date'] else None
    
    db.session.commit()
    return jsonify(task.to_dict())

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task."""
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return '', 204

# ========== INITIALIZATION ==========

def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
```''',
        category="coding",
        subcategory="python",
        tags=["flask", "rest-api", "python", "sqlalchemy", "crud", "backend"],
        source="local",
        created_at=datetime.now().isoformat()
    ))
    
    # bcrypt Authentication Pattern
    chunks.append(KnowledgeChunk(
        id="security_bcrypt_001",
        content='''bcrypt Password Hashing Best Practices:

```python
from flask_bcrypt import Bcrypt
from flask import Flask
import secrets

app = Flask(__name__)
bcrypt = Bcrypt(app)

# ========== PASSWORD HASHING ==========

def hash_password(plain_password: str) -> str:
    """
    Hash a password using bcrypt.
    
    bcrypt automatically:
    - Generates a random salt
    - Uses adaptive cost factor (default 12 rounds)
    - Produces a 60-character hash string
    """
    return bcrypt.generate_password_hash(plain_password).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.check_password_hash(hashed_password, plain_password)

# ========== USER MODEL WITH AUTH ==========

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password: str):
        """Hash and store password."""
        self.password_hash = hash_password(password)
    
    def check_password(self, password: str) -> bool:
        """Verify password."""
        return verify_password(password, self.password_hash)
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """Validate password meets security requirements."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if not any(c.isupper() for c in password):
            return False, "Password must contain uppercase letter"
        if not any(c.islower() for c in password):
            return False, "Password must contain lowercase letter"
        if not any(c.isdigit() for c in password):
            return False, "Password must contain a digit"
        return True, "Password is strong"

# ========== AUTH ROUTES ==========

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    # Validate
    if not email or '@' not in email:
        return jsonify({'error': 'Valid email required'}), 400
    
    valid, msg = User.validate_password_strength(password)
    if not valid:
        return jsonify({'error': msg}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    # Create user
    user = User(email=email)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created', 'id': user.id}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    user = User.query.filter_by(email=email).first()
    
    # Constant-time comparison to prevent timing attacks
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Generate session token (use JWT in production)
    token = secrets.token_urlsafe(32)
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user_id': user.id
    })
```

SECURITY NOTES:
- bcrypt cost factor of 12 is good for 2024 hardware
- Always use constant-time comparison for auth
- Store only the hash, never plain passwords
- Use HTTPS in production
- Implement rate limiting on auth endpoints
- Add account lockout after failed attempts''',
        category="security",
        subcategory="authentication",
        tags=["bcrypt", "password", "hashing", "authentication", "security", "python", "flask"],
        source="local",
        created_at=datetime.now().isoformat()
    ))
    
    # SQLite Schema Best Practices
    chunks.append(KnowledgeChunk(
        id="database_sqlite_001",
        content='''SQLite Database Schema Best Practices:

```sql
-- Enable foreign keys (must be done per connection)
PRAGMA foreign_keys = ON;

-- ========== USERS TABLE ==========
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);

-- ========== TASKS TABLE ==========
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT CHECK(priority IN ('low', 'medium', 'high')) DEFAULT 'medium',
    status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'cancelled')) DEFAULT 'pending',
    due_date DATE,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_tasks_user ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_due_date ON tasks(due_date);

-- ========== TRIGGER FOR updated_at ==========
CREATE TRIGGER update_tasks_timestamp 
    AFTER UPDATE ON tasks
BEGIN
    UPDATE tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_users_timestamp
    AFTER UPDATE ON users
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ========== USEFUL QUERIES ==========

-- Get user's tasks ordered by priority and due date
SELECT * FROM tasks 
WHERE user_id = ? AND status != 'completed'
ORDER BY 
    CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
    due_date ASC NULLS LAST;

-- Get overdue tasks
SELECT * FROM tasks
WHERE user_id = ? 
    AND status NOT IN ('completed', 'cancelled')
    AND due_date < DATE('now');

-- Task statistics
SELECT 
    status,
    COUNT(*) as count,
    priority
FROM tasks
WHERE user_id = ?
GROUP BY status, priority;
```

SQLITE TIPS:
- Use TEXT for strings (no VARCHAR in SQLite)
- BOOLEAN is stored as INTEGER (0/1)
- Use CHECK constraints for enums
- Create indexes on frequently queried columns
- Use AUTOINCREMENT for primary keys
- Enable foreign_keys pragma per connection
- Use triggers for updated_at timestamps''',
        category="database",
        subcategory="sqlite",
        tags=["sqlite", "sql", "database", "schema", "tasks", "best-practices"],
        source="local",
        created_at=datetime.now().isoformat()
    ))
    
    return chunks


def initialize_knowledge_base(embedding_fn: callable, data_dir: str = "data/knowledge"):
    """Initialize knowledge base with seed data."""
    kb = LocalKnowledgeBase(data_dir=data_dir, embedding_fn=embedding_fn)
    
    # Only seed if empty
    if len(kb.chunks) == 0:
        logger.info("[KnowledgeBase] Seeding with initial knowledge...")
        
        seed_chunks = create_seed_knowledge()
        for chunk in seed_chunks:
            kb.add_chunk(chunk)
        
        kb.save()
        logger.info(f"[KnowledgeBase] Seeded {len(seed_chunks)} knowledge chunks")
    
    return kb
