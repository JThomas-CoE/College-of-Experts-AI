
import re
import json
import time
import os
import sys
import argparse
import tempfile
from typing import *
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import ast
import urllib.request

from .config import *

# PHASE 5 — Template Store
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TemplateMatch:
    template_id:   str
    title:         str
    similarity:    float
    strength:      str           # "loose" | "medium" | "strong"
    scaffold_text: str           # formatted slot outline
    slot_sequence: List[str]     # slot IDs in order (TIER3 decomposition hint)
    category:      str = ""


class TemplateStore:
    """
    620-template store backed by config/framework_templates/all_templates.json.
    Embedding cosine similarity via always-resident mgr._embedder.
    Embeddings pre-computed at startup() — zero latency at query time.
    """

    def __init__(self) -> None:
        self._templates: List[dict] = []
        self._embeddings: Dict[str, np.ndarray] = {}  # template_id → L2-normalized vector

        if TEMPLATE_PATH.exists():
            try:
                self._templates = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
                cprint(f"[TemplateStore] Loaded {len(self._templates)} templates from {TEMPLATE_PATH.name}")
            except Exception as e:
                cprint(f"[TemplateStore] Could not load templates: {e}", "yellow")
        else:
            cprint(
                f"[TemplateStore] Template file not found: {TEMPLATE_PATH}\n"
                f"  Template matching will be disabled.",
                "yellow",
            )

    def startup(self, embedder: "EmbeddingManager") -> None:
        """Load template embeddings from cache (.npy) if up-to-date; compute+save otherwise.
        Cache stored at config/framework_templates/embedding_cache/.
        Invalidated automatically when all_templates.json changes (source hash check).
        """
        if not self._templates or embedder is None:
            cprint("[TemplateStore] Skipping — no templates or embedder.", "dim")
            return

        cache_dir = TEMPLATE_PATH.parent / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        template_file = str(TEMPLATE_PATH)

        # Try loading from cache via EmbeddingStore
        try:
            from embedding_manager import EmbeddingStore
            store = EmbeddingStore(store_path=cache_dir)
            if not store.needs_rebuild([template_file], EMBEDDING_MODEL, vector_name="template_vectors"):
                vecs, ids = store.load_embeddings("template_vectors")
                id_to_idx = {tid: i for i, tid in enumerate(ids)}
                loaded = 0
                for tpl in self._templates:
                    tid = tpl.get("id", "")
                    idx = id_to_idx.get(tid)
                    if idx is not None:
                        self._embeddings[tid] = vecs[idx]
                        loaded += 1
                if loaded == len(self._templates):
                    cprint(f"[TemplateStore] {loaded} embeddings loaded from cache (instant ✓)", "green")
                    return
                cprint(f"[TemplateStore] Cache partial ({loaded}/{len(self._templates)}) — rebuilding.", "dim")
                self._embeddings.clear()
        except Exception as e:
            cprint(f"[TemplateStore] Cache load skipped: {e}", "dim")
            store = None

        # Compute embeddings
        cprint(f"[TemplateStore] Computing embeddings for {len(self._templates)} templates…")
        t0 = time.time()
        texts = [f"{t.get('title', '')}. {t.get('description', '')}" for t in self._templates]
        try:
            batch_size = 96 if getattr(embedder, "device", "cpu") == "cpu" else 32
            if hasattr(embedder, "encode_batch"):
                vecs = embedder.encode_batch(texts, batch_size=batch_size, normalize=True)
            else:
                vecs = embedder.encode(texts, normalize=True)
            ids = []
            vec_list = []
            for i, tpl in enumerate(self._templates):
                tid = tpl.get("id", "")
                self._embeddings[tid] = vecs[i]
                ids.append(tid)
                vec_list.append(vecs[i])
            elapsed = time.time() - t0
            cprint(f"[TemplateStore] {len(self._embeddings)} embeddings computed ({elapsed:.1f}s)", "green")

            # Save to cache for next run
            try:
                import numpy as _np
                from embedding_manager import EmbeddingStore as _ES
                save_store = _ES(store_path=cache_dir)
                save_store.save_embeddings(
                    vectors=_np.stack(vec_list),
                    ids=ids,
                    model_name=EMBEDDING_MODEL,
                    source_files=[template_file],
                    vector_name="template_vectors",
                )
                cprint(f"[TemplateStore] Embeddings cached at {cache_dir}", "dim")
            except Exception as e:
                cprint(f"[TemplateStore] Cache save skipped: {e}", "dim")

        except Exception as e:
            cprint(f"[TemplateStore] Embedding computation failed: {e}", "yellow")

    def _cosine_best(
        self,
        query_text: str,
        embedder: "EmbeddingManager",
        category_filter: Optional[str] = None,
        category_boost: float = 0.05,
    ) -> Optional[Tuple[dict, float]]:
        """Return (template_dict, score) for best cosine match, or None if below TEMPLATE_LOOSE."""
        if not self._embeddings or embedder is None:
            return None
        try:
            qvec = embedder.encode(query_text, normalize=True)
            if qvec.ndim > 1:
                qvec = qvec[0]
        except Exception:
            return None

        best_score = -1.0
        best_tpl = None
        for tpl in self._templates:
            tid = tpl.get("id", "")
            vec = self._embeddings.get(tid)
            if vec is None:
                continue
            score = float(np.dot(qvec, vec))
            # Category boost for per-step matching
            if category_filter and tpl.get("category", "").lower() == category_filter.lower():
                score += category_boost
            if score > best_score:
                best_score = score
                best_tpl = tpl

        if best_score < TEMPLATE_LOOSE or best_tpl is None:
            return None
        return best_tpl, best_score

    def _build_match(self, tpl: dict, score: float) -> TemplateMatch:
        slots = tpl.get("slots", [])
        scaffold_lines = []
        for i, slot in enumerate(slots, 1):
            title = slot.get("title", slot.get("id", f"Step {i}"))
            desc = slot.get("description", "")
            scaffold_lines.append(f"Step {i} — {title}: {desc}" if desc else f"Step {i} — {title}")
        scaffold_text = "\n".join(scaffold_lines)
        slot_sequence = [s.get("id", f"step_{i}") for i, s in enumerate(slots, 1)]

        if score >= TEMPLATE_STRONG:
            strength = "strong"
        elif score >= TEMPLATE_MEDIUM:
            strength = "medium"
        else:
            strength = "loose"

        return TemplateMatch(
            template_id=tpl.get("id", ""),
            title=tpl.get("title", ""),
            similarity=score,
            strength=strength,
            scaffold_text=scaffold_text,
            slot_sequence=slot_sequence,
            category=tpl.get("category", ""),
        )

    def find_match(
        self, query: str, embedder: "EmbeddingManager"
    ) -> Optional[TemplateMatch]:
        result = self._cosine_best(query, embedder)
        if result is None:
            return None
        tpl, score = result
        return self._build_match(tpl, score)

    def find_match_for_step(
        self, step_desc: str, domain: str, embedder: "EmbeddingManager"
    ) -> Optional[TemplateMatch]:
        query_text = f"{domain}: {step_desc}"
        result = self._cosine_best(query_text, embedder, category_filter=domain)
        if result is None:
            return None
        tpl, score = result
        return self._build_match(tpl, score)


# ═══════════════════════════════════════════════════════════════════════════




