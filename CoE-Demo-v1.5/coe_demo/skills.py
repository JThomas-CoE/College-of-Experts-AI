
import json
import time
from typing import *
from dataclasses import dataclass
from pathlib import Path

from .config import *

# PHASE 8 — Skill Store
# ═══════════════════════════════════════════════════════════════════════════
# Skills are advisory reasoning guidance injected into the specialist system
# prompt.  They answer "how should I approach this problem?" whereas templates
# answer "what should the output look like?".
#
# Architectural role
# ──────────────────
#   Templates → injected into the USER prompt (structural scaffold, TIER3 slots)
#   Skills    → injected into the SPECIALIST SYSTEM prompt (advisory guidance)
#
# This distinction is intentional:
#   - Templates constrain the output shape — they belong with the task request.
#   - Skills inform specialist reasoning — they belong in the system context.
#
# Implementation contract
# ───────────────────────
# SkillStore is fully self-contained and requires only:
#   1. config/framework_skills/all_skills.json  — the skill library (committed)
#   2. An EmbeddingManager at startup() time     — same instance used by TemplateStore
#
# The embedding cache (skill_vectors.npy) is machine-specific and never committed.
# It is automatically rebuilt on first startup if absent or stale.
#
# Adding new skills
# ─────────────────
# Edit config/framework_skills/all_skills.json.  Each entry requires:
#   id          — unique snake_case identifier
#   domain      — "code" | "web" | "general" (used for domain-filter boost)
#   tags        — list of keyword strings (aids retrieval quality)
#   title       — short noun phrase (embedded as part of retrieval text)
#   description — sentence describing the problem area (embedded for retrieval)
#   guidance    — markdown bullet list injected into the specialist system prompt
#
# The retrieval text is:  "{title}. {description}"
# The injected text is:   guidance   (verbatim, prefixed by a heading line)
#
# Retrieval threshold
# ───────────────────
# SKILL_THRESHOLD = 0.50  (intentionally below TEMPLATE_LOOSE = 0.55)
# Skills are advisory — a looser match is acceptable.  When the top-1 skill
# scores above the threshold it is injected once; no strength tiers are needed
# because the injection wording is always advisory ("consider", not "must follow").
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SkillMatch:
    skill_id:    str
    title:       str
    domain:      str
    similarity:  float
    guidance:    str     # bullet-list text injected into specialist_system


class SkillStore:
    """
    Skill store backed by config/framework_skills/all_skills.json.
    Loaded once at startup; zero-latency retrieval via pre-computed cosine index.
    """

    def __init__(self) -> None:
        self._skills: List[dict] = []
        self._embeddings: Dict[str, Any] = {}  # skill_id → L2-normalised numpy vector

        if SKILL_PATH.exists():
            try:
                self._skills = json.loads(SKILL_PATH.read_text(encoding="utf-8"))
                cprint(f"[SkillStore] Loaded {len(self._skills)} skills from {SKILL_PATH.name}")
            except Exception as e:
                cprint(f"[SkillStore] Could not load skills: {e}", "yellow")
        else:
            cprint(
                f"[SkillStore] Skill file not found: {SKILL_PATH}\n"
                "  Skill-guided context will be disabled.",
                "yellow",
            )

    def startup(self, embedder: "EmbeddingManager") -> None:
        """Pre-compute (or load cached) skill embeddings.
        Cache stored at config/framework_skills/embedding_cache/.
        Invalidated automatically when all_skills.json changes.
        """
        if not self._skills or embedder is None:
            cprint("[SkillStore] Skipping — no skills or embedder.", "dim")
            return

        cache_dir = SKILL_PATH.parent / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        skill_file = str(SKILL_PATH)

        # Try loading from cache
        try:
            from embedding_manager import EmbeddingStore
            store = EmbeddingStore(store_path=cache_dir)
            if not store.needs_rebuild([skill_file], EMBEDDING_MODEL, vector_name="skill_vectors"):
                vecs, ids = store.load_embeddings("skill_vectors")
                id_to_idx = {sid: i for i, sid in enumerate(ids)}
                loaded = 0
                for sk in self._skills:
                    sid = sk.get("id", "")
                    idx = id_to_idx.get(sid)
                    if idx is not None:
                        self._embeddings[sid] = vecs[idx]
                        loaded += 1
                if loaded == len(self._skills):
                    cprint(f"[SkillStore] {loaded} embeddings loaded from cache (instant ✓)", "green")
                    return
                cprint(f"[SkillStore] Cache partial ({loaded}/{len(self._skills)}) — rebuilding.", "dim")
                self._embeddings.clear()
        except Exception as e:
            cprint(f"[SkillStore] Cache load skipped: {e}", "dim")
            store = None

        # Compute embeddings fresh
        cprint(f"[SkillStore] Computing embeddings for {len(self._skills)} skills…")
        t0 = time.time()
        texts = [f"{s.get('title', '')}. {s.get('description', '')}" for s in self._skills]
        try:
            batch_size = 96 if getattr(embedder, "device", "cpu") == "cpu" else 32
            if hasattr(embedder, "encode_batch"):
                vecs = embedder.encode_batch(texts, batch_size=batch_size, normalize=True)
            else:
                vecs = embedder.encode(texts, normalize=True)
            ids = []
            vec_list = []
            for i, sk in enumerate(self._skills):
                sid = sk.get("id", "")
                self._embeddings[sid] = vecs[i]
                ids.append(sid)
                vec_list.append(vecs[i])
            elapsed = time.time() - t0
            cprint(f"[SkillStore] {len(self._embeddings)} embeddings computed ({elapsed:.1f}s)", "green")

            # Save to cache
            try:
                import numpy as _np
                from embedding_manager import EmbeddingStore as _ES
                save_store = _ES(store_path=cache_dir)
                save_store.save_embeddings(
                    vectors=_np.stack(vec_list),
                    ids=ids,
                    model_name=EMBEDDING_MODEL,
                    source_files=[skill_file],
                    vector_name="skill_vectors",
                )
                cprint(f"[SkillStore] Embeddings cached at {cache_dir}", "dim")
            except Exception as e:
                cprint(f"[SkillStore] Cache save skipped: {e}", "dim")

        except Exception as e:
            cprint(f"[SkillStore] Embedding computation failed: {e}", "yellow")

    def find_match(
        self,
        query: str,
        domain: str,
        embedder: "EmbeddingManager",
    ) -> Optional[SkillMatch]:
        """Return the best matching SkillMatch for this query+domain, or None.

        Domain-matched skills receive a +0.05 cosine boost so that a code skill
        is preferred over an equally-scored general skill when answering a code
        query.  The retrieval threshold is intentionally loose (SKILL_THRESHOLD)
        because skills are advisory — a partial match is still useful guidance.
        """
        if not self._embeddings or embedder is None:
            return None
        try:
            qvec = embedder.encode(query, normalize=True)
            if qvec.ndim > 1:
                qvec = qvec[0]
        except Exception:
            return None

        best_score = -1.0
        best_skill = None
        domain_lower = (domain or "").lower()

        for sk in self._skills:
            sid = sk.get("id", "")
            vec = self._embeddings.get(sid)
            if vec is None:
                continue
            score = float(np.dot(qvec, vec))
            # Boost skills that belong to the active specialist domain
            if sk.get("domain", "").lower() == domain_lower:
                score += 0.05
            if score > best_score:
                best_score = score
                best_skill = sk

        if best_score < SKILL_THRESHOLD or best_skill is None:
            return None

        return SkillMatch(
            skill_id=best_skill.get("id", ""),
            title=best_skill.get("title", ""),
            domain=best_skill.get("domain", ""),
            similarity=best_score,
            guidance=best_skill.get("guidance", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════
