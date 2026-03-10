
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

# PHASE 7 — Knowledge Base Integration
# ═══════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION CONTRACT
# ────────────────
# KB_AVAILABLE is permanently False in this repository because the
# `knowledge_layer` package is an internal dependency that has not been
# open-sourced or published to PyPI.  All `kb_search()` calls silently return
# None and the pipeline continues without KB context.
#
# To implement a real knowledge base:
#   1. Create or install a package that exposes a `LocalKnowledgeBase` class
#      with at minimum:
#        __init__(db_path: Path, embedder)  — opens / creates the vector store
#        search(query: str, category: str, top_k: int) -> List[dict]
#             each result dict: {"text": str, "score": float, "metadata": dict}
#   2. Set KB_AVAILABLE = True in config.py (or make it a dynamic import test)
#   3. Import the class at the top of this file as `LocalKnowledgeBase`
#   4. No other changes are needed — init_kb() and kb_search() will activate
#      automatically.
#
# The stub below is intentionally minimal.  Do not add workarounds here;
# fix the KB package contract and flip KB_AVAILABLE instead.
# ═══════════════════════════════════════════════════════════════════════════

_kb_instance: Optional[object] = None
_kb_embedder_fn = None


def kb_search(query: str, domain: str) -> Optional[str]:
    global _kb_instance, _kb_embedder_fn
    if not KB_AVAILABLE or LocalKnowledgeBase is None:
        return None
    try:
        if _kb_instance is None:
            if _kb_embedder_fn is None:
                return None
            kb_dir = str(_HERE / "data" / "knowledge")
            _kb_instance = LocalKnowledgeBase(
                data_dir=kb_dir,
                embedding_fn=_kb_embedder_fn,
            )
        results = _kb_instance.search(query, category=domain, top_k=3)
        if results:
            return "\n\n".join(
                r[0].content if isinstance(r, tuple) else getattr(r, "content", str(r))
                for r in results[:3]
            )
    except Exception as e:
        cprint(f"[KB] Search error: {e}", "dim")
    return None


def kb_retrieve_pre(query: str, domain: str) -> Optional[str]:
    return kb_search(query, domain)


def kb_retrieve_retry(targeted_query: str, domain: str) -> Optional[str]:
    return kb_search(targeted_query, domain)


def init_kb(embedder: "EmbeddingManager") -> None:
    global _kb_embedder_fn
    if embedder and hasattr(embedder, "encode"):
        _kb_embedder_fn = embedder.encode
        kb_dir = _HERE / "data" / "knowledge"
        if not (kb_dir / "index.json").exists():
            cprint(
                f"[KB] Knowledge base not populated ({kb_dir}). "
                "KB-assisted context will be unavailable until the data directory is "
                "seeded. Run `python -m coe_demo.kb --seed` to initialise with "
                "built-in knowledge templates.",
                "yellow",
            )


# ═══════════════════════════════════════════════════════════════════════════




