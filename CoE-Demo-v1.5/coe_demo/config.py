
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


#!/usr/bin/env python3
"""
ollamaDemo.py — College of Experts Interactive CLI
================================================
Self-contained. All phases inline. No harness. No async. Proven OGA pattern only.
Device: DirectML (dml) by default for Radeon 890M iGPU (Ryzen AI 9 8060S).

Policy Source of Truth:
    ../ollamaDemo.md  ("V1 Policy Table (Source of Truth)")
This file should remain aligned with that policy table.

Usage:
    python ollamaDemo.py
    python ollamaDemo.py --device cpu --no-kb --no-enhance
    python ollamaDemo.py --no-confirm --session myproject
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure local src is on path (embedder, memory, KB)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from embedding_manager import EmbeddingManager
    EMBED_AVAILABLE = True
except ImportError:
    EMBED_AVAILABLE = False
    EmbeddingManager = None

try:
    from memory_backbone import MemoryBackbone, MemoryConfig
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    MemoryBackbone = None
    MemoryConfig = None

try:
    from knowledge_layer import LocalKnowledgeBase
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False
    LocalKnowledgeBase = None


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"
CONFIG_PATH = _HERE / "config" / "demo_config.json"

# Template file — look in local config first, then parent workspace config
_TEMPLATE_CANDIDATES = [
    _HERE / "config" / "framework_templates" / "all_templates.json",
    _HERE.parent / "config" / "framework_templates" / "all_templates.json",
]
TEMPLATE_PATH = next((p for p in _TEMPLATE_CANDIDATES if p.exists()), _TEMPLATE_CANDIDATES[0])

# Skill file — advisory reasoning guidance injected into specialist system prompts
_SKILL_CANDIDATES = [
    _HERE / "config" / "framework_skills" / "all_skills.json",
    _HERE.parent / "config" / "framework_skills" / "all_skills.json",
]
SKILL_PATH = next((p for p in _SKILL_CANDIDATES if p.exists()), _SKILL_CANDIDATES[0])
SKILL_THRESHOLD = 0.50  # intentionally below TEMPLATE_LOOSE (0.55) — skills are advisory

EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_CONFIG: Dict = {
    "device":        "dml",
    "enhance":       True,
    "confirm":       True,
    "kb":            True,
    "self_grade":    True,
    "session":       None,
    "render_mode":   "intermediate+final",  # or "final only"
    "embed_cpu_batch_size": 96,
    "embed_cpu_threads": 12,
    "max_runtime_seq_tokens": 16384,
    "model_base_dir": None,   # override search dirs (e.g. "D:/models")
}

# Directories searched in order when resolving model folder names.
# The config 'model_base_dir' setting prepends an additional entry at runtime.
_DEFAULT_MODEL_SEARCH_DIRS: List[str] = [
    "D:/models",
    "C:/models",
    str(_HERE / "models"),
    str(Path(_HERE).parent / "models"),
]

def resolve_model_path(name: str, extra_dirs: Optional[List[str]] = None) -> str:
    """Return the first existing directory matching *name* across search dirs.
    Falls back to *name* as-is (absolute path or relative) if nothing is found.
    """
    if Path(name).is_absolute():
        return name  # caller provided a full path
    search = list(extra_dirs or []) + _DEFAULT_MODEL_SEARCH_DIRS
    for base in search:
        candidate = Path(base) / name
        if candidate.exists():
            return str(candidate)
    # Last-ditch: try relative to cwd
    if Path(name).exists():
        return str(Path(name).resolve())
    return name  # not found — let the caller surface the error

# Bare folder names only — resolve_model_path() maps these to real paths at use time.
DOMAIN_MODELS: Dict[str, str] = {
    "code":       "ollama://CoE-python2-40b-A3b:q4_K_M",
    "web":        "ollama://CoE-WEB2-40b-A3b:q4_K_M",
    "supervisor": "Nanbeige4.1-3B-ONNX-INT4",
}

# Temporarily disable known-bad specialist models and fall back to synthetic
# supervisor-driven specialists instead. This preserves domain personas without
# loading unstable domain models.
DISABLED_SPECIALIST_DOMAINS = {"medical", "legal", "math", "sql"}

TEMPLATE_STRONG = 0.85
TEMPLATE_MEDIUM = 0.70
TEMPLATE_LOOSE  = 0.55

TEMPLATE_IMPERATIVE = {
    "strong": "A closely matching structural template is available — follow this structure:",
    "medium": "A related structural template is available — follow its phase structure where it applies to this task:",
    "loose":  "A loosely related structural template is available — treat as optional inspiration, adapt freely:",
}

FAIL_CATEGORIES = ["FAIL_INCOMPLETE", "FAIL_OFFTOPIC", "FAIL_FORMAT"]
SPECIALIST_SELF_GRADE_DOMAINS = {"code", "math", "web"}

console = Console() if RICH_AVAILABLE else None

DEBUG_TRACE_ENABLED = False
DEBUG_TRACE_PATH = _HERE / "acceptance_runs" / "debug_trace.jsonl"


def set_debug_trace(enabled: bool, trace_path: Optional[Path] = None) -> None:
    global DEBUG_TRACE_ENABLED, DEBUG_TRACE_PATH
    DEBUG_TRACE_ENABLED = bool(enabled)
    if trace_path is not None:
        DEBUG_TRACE_PATH = trace_path


def debug_log(event: str, **fields) -> None:
    if not DEBUG_TRACE_ENABLED:
        return
    try:
        DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "event": event,
            **fields,
        }
        with DEBUG_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
    except Exception:
        pass


def cprint(msg: str, style: str = "") -> None:
    if console:
        console.print(msg, style=style) if style else console.print(msg)
    else:
        print(msg)


def terminal_friendly_text(text: str) -> str:
    """Reduce markdown/LaTeX noise for terminal display."""
    out = text or ""
    # Block/inline math delimiters
    out = re.sub(r"\$\$(.*?)\$\$", lambda m: "\n" + m.group(1).strip() + "\n", out, flags=re.DOTALL)
    out = re.sub(r"\$(.+?)\$", lambda m: m.group(1), out, flags=re.DOTALL)
    # Common LaTeX commands/operators
    replacements = {
        r"\\frac": "frac",
        r"\\cdot": "*",
        r"\\times": "×",
        r"\\leq": "<=",
        r"\\geq": ">=",
        r"\\neq": "!=",
        r"\\approx": "≈",
        r"\\to": "->",
        r"\\rightarrow": "->",
        r"\\left": "",
        r"\\right": "",
        r"\\sin": "sin",
        r"\\cos": "cos",
        r"\\tan": "tan",
        r"\\log": "log",
        r"\\ln": "ln",
        r"\\int": "integral",
        r"\\sum": "sum",
        r"\\sqrt": "sqrt",
        r"\\pi": "pi",
    }
    for src, dst in replacements.items():
        out = re.sub(src, dst, out)
    out = out.replace("\\(", "").replace("\\)", "")
    out = out.replace("\\[", "").replace("\\]", "")
    out = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", out)
    out = re.sub(r"frac\s*\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", out)
    return out


def render_response(text: str, title: str = "Response") -> None:
    """Render copy-paste-safe response text (no box borders/markdown transforms)."""
    text = terminal_friendly_text(text)
    if RICH_AVAILABLE and console:
        console.print(f"\n[bold green]{title}[/bold green]")
        console.print(Text(text or "", no_wrap=False))
    else:
        print("\n" + "─" * 60)
        print(text)
        print("─" * 60)


def normalize_render_mode(value: Optional[str]) -> str:
    """Normalize render mode to one of: intermediate+final, final only."""
    raw = (value or "").strip().lower()
    if raw in ("final", "final-only", "final_only", "final only"):
        return "final only"
    if raw in ("intermediate", "both", "all", "intermediate+final", "default", ""):
        return "intermediate+final"
    return "intermediate+final"


def is_show_intermediate_outputs(settings: dict) -> bool:
    return normalize_render_mode(settings.get("render_mode")) != "final only"


def is_self_grading_enabled(settings: dict) -> bool:
    return bool(settings.get("self_grade", True))


# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_settings(cli_args) -> Tuple[Dict, Dict]:
    """Load settings from config file, override with CLI args. Returns (settings, sources)."""
    settings = DEFAULT_CONFIG.copy()
    sources = {k: "default" for k in settings}

    # Load or create config.json
    if CONFIG_PATH.exists():
        try:
            file_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in file_cfg.items():
                if k in settings:
                    settings[k] = v
                    sources[k] = "config.json"
        except Exception as e:
            cprint(f"[CONFIG] Warning: could not parse {CONFIG_PATH}: {e}", "yellow")
    else:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        cprint(f"[CONFIG] Created default config at {CONFIG_PATH}")

    # CLI overrides
    if hasattr(cli_args, "device") and cli_args.device != "dml":
        settings["device"] = cli_args.device
        sources["device"] = "cli-arg"
    if getattr(cli_args, "no_kb", False):
        settings["kb"] = False
        sources["kb"] = "cli-arg"
    if getattr(cli_args, "no_enhance", False):
        settings["enhance"] = False
        sources["enhance"] = "cli-arg"
    if getattr(cli_args, "no_confirm", False):
        settings["confirm"] = False
        sources["confirm"] = "cli-arg"
    if getattr(cli_args, "no_self_grade", False):
        settings["self_grade"] = False
        sources["self_grade"] = "cli-arg"
    if getattr(cli_args, "session", None):
        settings["session"] = cli_args.session
        sources["session"] = "cli-arg"
    if getattr(cli_args, "embed_cpu_batch_size", None):
        settings["embed_cpu_batch_size"] = max(8, int(cli_args.embed_cpu_batch_size))
        sources["embed_cpu_batch_size"] = "cli-arg"
    if getattr(cli_args, "embed_cpu_threads", None):
        settings["embed_cpu_threads"] = max(1, int(cli_args.embed_cpu_threads))
        sources["embed_cpu_threads"] = "cli-arg"

    return settings, sources


def save_setting(key: str, value) -> None:
    """Write a single setting back to config/demo_config.json."""
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg[key] = value
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════






