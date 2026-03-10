
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

_HERE = Path(__file__).parent

# PHASE 4 — Session Memory Store
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationArtifact:
    artifact_id: str
    task_key: str
    phase: str
    domain: str
    verdict: str
    text: str
    token_estimate: int
    created_at: str


SESSION_RAW_CONTEXT_TASKS = 2
SESSION_ARCHIVE_AFTER_TASKS = 2
SESSION_ARCHIVE_QUERY_TOP_K = 3
SESSION_MAX_ACCEPTED_TEXT_CHARS = 24000
SESSION_ARCHIVE_SOURCE_MAX_CHARS = 12000
SESSION_ARCHIVE_MIN_SOURCE_CHARS = 250
SESSION_ARCHIVE_SUMMARY_MAX_TOKENS = 220
SESSION_MAX_ARCHIVED_MEMORIES = 128


@dataclass
class ArchivedMemory:
    archive_key: str
    task_key: str
    query: str
    domains: List[str]
    summary_text: str
    source_char_count: int
    created_at: str
    vector: Optional[List[float]] = None


class GenerationArtifactStore:
    """In-RAM artifact store for accepted generation outputs by phase."""

    def __init__(self, max_items: int = 512):
        self._max_items = max(64, int(max_items))
        self._items: Dict[str, GenerationArtifact] = {}
        self._order: List[str] = []
        self._by_task: Dict[str, List[str]] = {}

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\S+", text))

    def put(self, task_key: str, phase: str, text: str, domain: str = "", verdict: str = "") -> str:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_suffix = task_key.split(":")[-1]
        phase_slug = re.sub(r"[^a-z0-9]+", "_", (phase or "phase").lower()).strip("_") or "phase"
        artifact_id = f"{task_suffix}:{phase_slug}:{len(self._order):03d}"
        item = GenerationArtifact(
            artifact_id=artifact_id,
            task_key=task_key,
            phase=phase,
            domain=domain,
            verdict=verdict,
            text=text or "",
            token_estimate=self._estimate_tokens(text or ""),
            created_at=created_at,
        )
        self._items[artifact_id] = item
        self._order.append(artifact_id)
        self._by_task.setdefault(task_key, []).append(artifact_id)

        while len(self._order) > self._max_items:
            drop_id = self._order.pop(0)
            old = self._items.pop(drop_id, None)
            if old is not None:
                task_list = self._by_task.get(old.task_key, [])
                self._by_task[old.task_key] = [aid for aid in task_list if aid != drop_id]
                if not self._by_task[old.task_key]:
                    self._by_task.pop(old.task_key, None)

        return artifact_id

    def get_text(self, artifact_id: str) -> str:
        item = self._items.get(artifact_id)
        return item.text if item else ""

    def get(self, artifact_id: str) -> Optional[GenerationArtifact]:
        return self._items.get(artifact_id)

    def list_task_artifacts(self, task_key: str) -> List[GenerationArtifact]:
        ids = self._by_task.get(task_key, [])
        return [self._items[i] for i in ids if i in self._items]

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()
        self._by_task.clear()

class SessionStore:
    """Hot raw session memory + compressed searchable archive persisted in SQLite."""

    def __init__(self, session_prefix: Optional[str] = None):
        self.session_id = (
            f"{session_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if session_prefix
            else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.task_seq = 0
        self._artifacts = GenerationArtifactStore()
        self._recent_tasks: List[dict] = []
        self._pending_switch: Optional[dict] = None
        self._db: Optional["MemoryBackbone"] = None
        self._archive_cache: List[ArchivedMemory] = []
        self._archived_task_keys: set = set()
        self._archive_seq: int = 0
        self._archive_loaded = False

        if MEMORY_AVAILABLE:
            db_path = _HERE / "data" / "memory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._db = MemoryBackbone(MemoryConfig(db_path=db_path))
            except Exception as e:
                cprint(f"[SessionStore] Could not init DB: {e}", "yellow")

    def _load_archived_memories(self) -> None:
        if self._archive_loaded:
            return
        self._archive_loaded = True
        if not self._db:
            return
        try:
            entries = self._db.search(f"{self.session_id}:archive:", tier="semantic", limit=SESSION_MAX_ARCHIVED_MEMORIES)
            loaded: List[ArchivedMemory] = []
            max_seq = -1
            for entry in entries:
                value = entry.value if isinstance(entry.value, dict) else {}
                archive_key = str(value.get("archive_key") or entry.key)
                task_key = str(value.get("task_key") or "")
                seq_match = re.search(r":archive:(\d+)$", archive_key)
                if seq_match:
                    max_seq = max(max_seq, int(seq_match.group(1)))
                loaded.append(ArchivedMemory(
                    archive_key=archive_key,
                    task_key=task_key,
                    query=str(value.get("query") or ""),
                    domains=[str(d).lower() for d in (value.get("domains") or [])],
                    summary_text=str(value.get("summary_text") or ""),
                    source_char_count=int(value.get("source_char_count") or 0),
                    created_at=str(value.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    vector=value.get("vector") if isinstance(value.get("vector"), list) else None,
                ))
                if task_key:
                    self._archived_task_keys.add(task_key)
            self._archive_cache = loaded
            self._archive_seq = max_seq + 1
        except Exception as e:
            cprint(f"[SessionStore] Archive load warning: {e}", "dim")

    def peek_task_key(self) -> str:
        return f"{self.session_id}:task:{self.task_seq:04d}"

    def add_generation_artifact(
        self,
        task_key: str,
        phase: str,
        text: str,
        domain: str = "",
        verdict: str = "",
    ) -> str:
        return self._artifacts.put(task_key, phase, text, domain=domain, verdict=verdict)

    def get_generation_text(self, artifact_id: str) -> str:
        return self._artifacts.get_text(artifact_id)

    def list_generation_artifacts(self, task_key: str) -> List[dict]:
        out: List[dict] = []
        for item in self._artifacts.list_task_artifacts(task_key):
            out.append({
                "id": item.artifact_id,
                "phase": item.phase,
                "domain": item.domain,
                "verdict": item.verdict,
                "token_estimate": item.token_estimate,
                "text_snippet": item.text[:120],
                "created_at": item.created_at,
            })
        return out

    def _accepted_text_for_task(self, task: dict, output: str = "") -> str:
        if not isinstance(task, dict):
            return (output or "").strip()
        accepted_text = str(task.get("accepted_text") or "")
        arts = task.get("artifacts", {}) or {}
        final_id = arts.get("final_artifact_id") if isinstance(arts, dict) else None
        if final_id:
            artifact_text = self.get_generation_text(final_id)
            if artifact_text:
                accepted_text = artifact_text
        if not accepted_text:
            accepted_text = output or str(task.get("output_snippet") or "")
        return accepted_text.strip()

    def write_task(
        self,
        query: str,
        tier: str,
        domains: List[str],
        output: str,
        artifacts: dict,
    ) -> None:
        key = f"{self.session_id}:task:{self.task_seq:04d}"
        accepted_text = self._accepted_text_for_task({"artifacts": artifacts}, output)
        value = {
            "task_key": key,
            "query": query,
            "tier": tier,
            "domains": domains,
            "output_snippet": output[:300],
            "accepted_text": accepted_text[:SESSION_MAX_ACCEPTED_TEXT_CHARS],
            "accepted_text_truncated": len(accepted_text) > SESSION_MAX_ACCEPTED_TEXT_CHARS,
            "artifacts": artifacts,
        }
        meta = {
            "tier": tier,
            "domains": domains,
            "query_snippet": query[:80],
            "artifact_keys": list(artifacts.keys()),
        }
        if self._db:
            try:
                self._db.write(key, value, tier="episodic", metadata=meta)
            except Exception as e:
                cprint(f"[SessionStore] Write error: {e}", "dim")
        self._recent_tasks.insert(0, value)
        if len(self._recent_tasks) > 50:
            self._recent_tasks = self._recent_tasks[:50]
        self.task_seq += 1

    def _fit_archive_source(self, text: str, max_chars: int = SESSION_ARCHIVE_SOURCE_MAX_CHARS) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        head = max_chars * 2 // 3
        tail = max_chars - head - 32
        return text[:head] + "\n\n...[middle omitted for memory compression]...\n\n" + text[-max(0, tail):]

    def _archive_summary_fallback(self, task: dict, source_text: str) -> str:
        domains = ", ".join(task.get("domains", [])) or "supervisor"
        query = str(task.get("query") or "")
        excerpt = self._fit_archive_source(source_text, max_chars=900)
        return (
            f"TASK: {query}\n"
            f"DOMAINS: {domains}\n"
            f"RESULT: Prior accepted output related to this task was stored for follow-up continuity.\n"
            f"FACTS: {excerpt}"
        )

    def _summarize_for_archive(self, task: dict, source_text: str, mgr: Optional["ModelManager"]) -> str:
        if not mgr:
            return self._archive_summary_fallback(task, source_text)
        query = str(task.get("query") or "")
        domains = ", ".join(task.get("domains", [])) or "supervisor"
        fitted_source = self._fit_archive_source(source_text)
        prompt = (
            "Compress this prior task into durable searchable memory. Do not include chain-of-thought.\n"
            "Return plain text with these exact headings:\n"
            "TASK:\nRESULT:\nFACTS:\nDECISIONS:\nCONSTRAINTS:\nOPEN_ISSUES:\n\n"
            f"Original query: {query}\n"
            f"Domains: {domains}\n\n"
            f"Accepted output:\n{fitted_source}"
        )
        system = (
            "You are a memory consolidation engine. Produce concise plain-text archival memory for later retrieval. "
            "Keep concrete facts, decisions, constraints, and unresolved issues. Do not output analysis or think blocks."
        )
        try:
            summary = mgr.generate_supervisor(
                prompt,
                system=system,
                max_tokens=SESSION_ARCHIVE_SUMMARY_MAX_TOKENS,
                temperature=0.1,
                think_budget=0,
                disable_thinking=True,
            ).strip()
            summary = strip_think(summary).strip()
            if len(summary) < 60 or "TASK:" not in summary:
                return self._archive_summary_fallback(task, source_text)
            return summary
        except Exception:
            return self._archive_summary_fallback(task, source_text)

    def _lexical_memory_score(self, query: str, text: str) -> float:
        q_terms = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
        if not q_terms:
            return 0.0
        hay = (text or "").lower()
        hits = sum(1 for token in q_terms if token in hay)
        return hits / max(1, len(q_terms))

    def _search_archived_memories(
        self,
        query: str,
        embedder: Optional["EmbeddingManager"] = None,
        limit: int = SESSION_ARCHIVE_QUERY_TOP_K,
    ) -> List[ArchivedMemory]:
        self._load_archived_memories()
        if not self._archive_cache:
            return []

        qvec = None
        if embedder is not None:
            try:
                qvec = np.asarray(embedder.encode(query, normalize=True), dtype=np.float32)
            except Exception:
                qvec = None

        scored: List[Tuple[float, ArchivedMemory]] = []
        for item in self._archive_cache:
            score = 0.0
            if qvec is not None and item.vector:
                try:
                    ivec = np.asarray(item.vector, dtype=np.float32)
                    if ivec.shape == qvec.shape:
                        score = float(np.dot(qvec, ivec))
                except Exception:
                    score = 0.0
            if score <= 0.0:
                score = self._lexical_memory_score(query, item.summary_text)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[ArchivedMemory] = []
        for score, item in scored:
            if qvec is not None and score < 0.18:
                continue
            if qvec is None and score <= 0.0:
                continue
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def maybe_archive_history(
        self,
        mgr: Optional["ModelManager"] = None,
        embedder: Optional["EmbeddingManager"] = None,
    ) -> None:
        self._load_archived_memories()
        tasks = self.get_recent_tasks(12)
        if len(tasks) <= SESSION_ARCHIVE_AFTER_TASKS:
            return

        for task in reversed(tasks[SESSION_ARCHIVE_AFTER_TASKS:]):
            task_key = str(task.get("task_key") or "")
            if not task_key or task_key in self._archived_task_keys:
                continue
            source_text = self._accepted_text_for_task(task)
            if len(source_text) < SESSION_ARCHIVE_MIN_SOURCE_CHARS:
                continue

            summary_text = self._summarize_for_archive(task, source_text, mgr)
            vector = None
            if embedder is not None:
                try:
                    vector = np.asarray(embedder.encode(summary_text, normalize=True), dtype=np.float32).tolist()
                except Exception:
                    vector = None

            archive_key = f"{self.session_id}:archive:{self._archive_seq:04d}"
            self._archive_seq += 1
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = ArchivedMemory(
                archive_key=archive_key,
                task_key=task_key,
                query=str(task.get("query") or ""),
                domains=[str(d).lower() for d in (task.get("domains") or [])],
                summary_text=summary_text,
                source_char_count=len(source_text),
                created_at=created_at,
                vector=vector,
            )
            self._archive_cache.insert(0, entry)
            self._archived_task_keys.add(task_key)

            if self._db:
                try:
                    self._db.write(
                        archive_key,
                        {
                            "archive_key": archive_key,
                            "task_key": task_key,
                            "query": entry.query,
                            "domains": entry.domains,
                            "summary_text": entry.summary_text,
                            "source_char_count": entry.source_char_count,
                            "created_at": entry.created_at,
                            "vector": entry.vector,
                        },
                        tier="semantic",
                        metadata={"type": "archived_memory", "session_id": self.session_id},
                    )
                except Exception as e:
                    cprint(f"[SessionStore] Archive write warning: {e}", "dim")

            cprint(f"[Memory] Archived {task_key.split(':')[-1]} for searchable recall.", "dim")
            break

    def get_context_for_query(
        self,
        query: str,
        embedder: Optional["EmbeddingManager"] = None,
    ) -> str:
        try:
            recent = self.get_recent_tasks(SESSION_RAW_CONTEXT_TASKS)
            archived = self._search_archived_memories(query, embedder, SESSION_ARCHIVE_QUERY_TOP_K)
            if not recent and not archived:
                return ""

            lines: List[str] = []
            if recent:
                lines.append(f"=== Session Context (last {len(recent)} tasks) ===")
                for v in recent:
                    t = v.get("tier", "?")
                    dom = ",".join(v.get("domains", []))
                    qsnip = v.get("query", "")[:120]
                    accepted = self._accepted_text_for_task(v)
                    lines.append(f"[{t} | {dom}] Query: {qsnip}")
                    if accepted:
                        lines.append(f"Accepted final excerpt: {accepted[:320]}")
                lines.append("===")

            if archived:
                lines.append("=== Relevant Archived Memory ===")
                for item in archived:
                    dom = ",".join(item.domains) or "supervisor"
                    lines.append(f"[ARCHIVE | {dom}] Query: {item.query[:120]}")
                    lines.append(item.summary_text[:520])
                lines.append("===")

            return "\n".join(lines)
        except Exception:
            return ""

    def get_recent_tasks(self, n: int = 5) -> List[dict]:
        if self._db:
            try:
                entries = self._db.search(self.session_id, tier="episodic", limit=n)
                vals = [e.value if isinstance(e.value, dict) else {} for e in entries]
                if vals:
                    return vals
            except Exception:
                pass
        try:
            return self._recent_tasks[: max(0, int(n))]
        except Exception:
            return []

    def get_recent_specialist_tasks(self, n: int = 5) -> List[dict]:
        """Like get_recent_tasks() but returns only TIER2+ specialist tasks.

        TIER1 tasks produce conversational supervisor text (greetings, meta answers,
        explanations) that carry no injectable code/content artifact.  Any caller
        that wants a *prior specialist artifact* to continue — e.g. the follow-up
        context injection in run_tier2 — must use this method so that tier-blind
        contamination is impossible by construction, not just by runtime guard.
        """
        # Over-fetch then filter; TIER1 tasks are minority so 4× is safe.
        candidates = self.get_recent_tasks(max(n * 4, 20))
        result: List[dict] = []
        for t in candidates:
            tier_str = str(t.get("tier", "")).upper()
            domains = [str(d).lower() for d in (t.get("domains") or [])]
            if tier_str.startswith("TIER1") or domains == ["supervisor"]:
                continue
            result.append(t)
            if len(result) >= n:
                break
        return result

    def set_pending_switch(
        self,
        query: str,
        from_domain: Optional[str],
        to_domain: Optional[str],
        interpretation: str = "",
    ) -> None:
        self._pending_switch = {
            "query": query,
            "from_domain": from_domain,
            "to_domain": to_domain,
            "interpretation": interpretation,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_pending_switch(self) -> Optional[dict]:
        return dict(self._pending_switch) if self._pending_switch else None

    def clear_pending_switch(self) -> None:
        self._pending_switch = None

    def clear_session(self) -> None:
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.task_seq = 0
        self._artifacts.clear()
        self._recent_tasks.clear()
        self._pending_switch = None
        self._archive_cache.clear()
        self._archived_task_keys.clear()
        self._archive_seq = 0
        self._archive_loaded = False

    def get_stats(self) -> dict:
        self._load_archived_memories()
        return {
            "session_id": self.session_id,
            "task_count": self.task_seq,
            "artifact_count": self._artifacts.count(),
            "archive_count": len(self._archive_cache),
            "pending_switch": self._pending_switch.get("to_domain") if self._pending_switch else None,
            "db_available": self._db is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════




