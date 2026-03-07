#!/usr/bin/env python3
"""
demo.py — College of Experts Interactive CLI
=============================================
Self-contained. All phases inline. No harness. No async. Proven OGA pattern only.
Device: DirectML (dml) by default for Radeon 890M iGPU (Ryzen AI 9 8060S).

Policy Source of Truth:
    ./demo.md  ("V1 Policy Table (Source of Truth)")
This file should remain aligned with that policy table.

Usage:
    python demo.py
    python demo.py --device cpu --no-kb --no-enhance
    python demo.py --no-confirm --session myproject
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
INFERENCE_WORKER_PATH = _HERE / "inference_worker.py"

# Template file — look in local config first, then parent workspace config
_TEMPLATE_CANDIDATES = [
    _HERE / "config" / "framework_templates" / "all_templates.json",
    _HERE.parent / "config" / "framework_templates" / "all_templates.json",
]
TEMPLATE_PATH = next((p for p in _TEMPLATE_CANDIDATES if p.exists()), _TEMPLATE_CANDIDATES[0])

EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_CONFIG: Dict = {
    "device":        "dml",
    "enhance":       True,
    "confirm":       True,
    "kb":            True,
    "session":       None,
    "render_mode":   "intermediate+final",  # or "final only"
    "embed_cpu_batch_size": 96,
    "embed_cpu_threads": 12,
    "max_runtime_seq_tokens": 4096,
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
    "code":       "Qwen2.5-Coder-7B-DML",
    "sql":        "sqlcoder-7b-2-DML",
    "math":       "Qwen2.5-Math-7B-DML",
    "medical":    "BioMistral-7B-DML",
    "legal":      "law-LLM-DML",
    "web":        "Qwen2.5-Coder-7B-DML",
    "supervisor": "Nanbeige4.1-3B-ONNX-INT4",
}

# Temporarily disable known-bad specialist models and fall back to synthetic
# supervisor-driven specialists instead. This preserves domain personas without
# loading unstable domain models.
DISABLED_SPECIALIST_DOMAINS = {"medical", "legal"}

TEMPLATE_STRONG = 0.85
TEMPLATE_MEDIUM = 0.70
TEMPLATE_LOOSE  = 0.55

TEMPLATE_IMPERATIVE = {
    "strong": "A closely matching structural template is available — follow this structure:",
    "medium": "A related structural template is available — follow its phase structure where it applies to this task:",
    "loose":  "A loosely related structural template is available — treat as optional inspiration, adapt freely:",
}

FAIL_CATEGORIES = ["FAIL_INCOMPLETE", "FAIL_OFFTOPIC", "FAIL_FORMAT"]
SPECIALIST_SELF_GRADE_DOMAINS = {"code", "math"}

console = Console() if RICH_AVAILABLE else None

_DEBUG_TRACE_ENABLED = False
_DEBUG_TRACE_PATH = _HERE / "acceptance_runs" / "debug_trace.jsonl"


def set_debug_trace(enabled: bool, trace_path: Optional[Path] = None) -> None:
    global _DEBUG_TRACE_ENABLED, _DEBUG_TRACE_PATH
    _DEBUG_TRACE_ENABLED = bool(enabled)
    if trace_path is not None:
        _DEBUG_TRACE_PATH = trace_path


def debug_log(event: str, **fields) -> None:
    if not _DEBUG_TRACE_ENABLED:
        return
    try:
        _DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "event": event,
            **fields,
        }
        with _DEBUG_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
    except Exception:
        pass


def cprint(msg: str, style: str = "") -> None:
    if console:
        console.print(msg, style=style) if style else console.print(msg)
    else:
        print(msg)


def _terminal_friendly_text(text: str) -> str:
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
    text = _terminal_friendly_text(text)
    if RICH_AVAILABLE and console:
        console.print(f"\n[bold green]{title}[/bold green]")
        console.print(Text(text or "", no_wrap=False))
    else:
        print("\n" + "─" * 60)
        print(text)
        print("─" * 60)


def _normalize_render_mode(value: Optional[str]) -> str:
    """Normalize render mode to one of: intermediate+final, final only."""
    raw = (value or "").strip().lower()
    if raw in ("final", "final-only", "final_only", "final only"):
        return "final only"
    if raw in ("intermediate", "both", "all", "intermediate+final", "default", ""):
        return "intermediate+final"
    return "intermediate+final"


def _show_intermediate_outputs(settings: dict) -> bool:
    return _normalize_render_mode(settings.get("render_mode")) != "final only"


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
# PHASE 1 — Core Inference Engine: ModelManager
# ═══════════════════════════════════════════════════════════════════════════

# V1 policy constants (documentation/alignment block)
# Keep aligned with ./demo.md "V1 Policy Table (Source of Truth)".
POLICY_V1_LIFECYCLE = "domain-sticky-specialist"          # keep specialist until domain changes
POLICY_V1_TIER2_CONTRACT = "benchmark-v4"                 # follow run_coe_benchmark_v4 TIER2 contract
POLICY_V1_TIER3_SYNTHESIS_DEFAULT = "enabled"             # TIER3 final synthesis on by default
POLICY_V1_RENDER_DEFAULT = "intermediate+final"           # configurable alternative: "final only"
POLICY_V1_GRADING_RETRY = "benchmark-v4-plus-deterministic-signal"

# Per-tier thinking budgets (Phase 1 token cap in _generate_with).
# Phase 1 generates up to this many tokens watching for </think>; if the
# budget runs out, the block is force-closed so Phase 2 produces the answer.
THINK_BUDGET_T1   =  100   # TIER1  — trivial queries, minimal reasoning
THINK_BUDGET_T2   =  700   # TIER2  — single domain task
THINK_BUDGET_T3   = 2000   # TIER3  — complex multi-domain, per-step
THINK_BUDGET_UTIL =  100   # utility — classification, KB check, grading

# Per-tier output budgets (Phase 2 / single-phase token cap).
# T2/T3 budgets apply to primary (zero-shot) generation AND the single retry
# while the specialist is still in VRAM.  Grading and synthesis are separate
# processes with their own budgets and never use the specialist.
OUTPUT_BUDGET_T1         = 1024   # TIER1  — concise direct answer
OUTPUT_BUDGET_T2         = 4096   # TIER2  — primary generation + retry
OUTPUT_BUDGET_T3         = 4096   # TIER3  — per-step primary generation + retry
OUTPUT_BUDGET_SYNTHESIS  = 4096   # supervisor synthesis/refinement pass (all tiers)
SYNTHESIS_DRAFT_TOKEN_CAP = 2048  # safety cap for synthesis input draft payload
RETRY_DRAFT_TOKEN_CAP     = 1024  # cap prior-draft payload in retry prompts
ENABLE_TIER2_SYNTHESIS = True     # v1 benchmark-style contract: synthesize then keep better answer
KEEP_SPECIALIST_WARM = True       # keep specialist resident across queries for stability/latency

EMPTY_RESPONSE_MESSAGE = (
    "Sorry — the models were not able to generate an acceptable response for this request. "
    "Please try rephrasing your prompt or splitting it into smaller steps."
)

# Folder-name substrings that identify thinking models (case-insensitive).
_THINKING_MODEL_MARKERS = ("nanbeige", "deepseek-r1", "qwq", "-r1")

def _is_thinking_model(path: str) -> bool:
    """Return True if the model is known to emit <think>...</think> blocks."""
    p = path.lower()
    return any(m in p for m in _THINKING_MODEL_MARKERS)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models.
    If the block is unterminated (budget ran out mid-think), discard everything
    from <think> onward so callers never see raw reasoning tokens.
    """
    text = _THINK_BLOCK.sub("", text)
    open_idx = text.lower().find("<think>")
    if open_idx != -1:
        text = text[:open_idx]
    return text


def _safe_first_line(text: str, default: str = "") -> str:
    """Return first non-empty stripped line; never raises on empty input."""
    s = (text or "").strip()
    if not s:
        return default
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return default


def _normalize_grade_verdict(raw: str) -> Tuple[str, str]:
    """Return (verdict_word, verdict_line) from noisy grader output.

    verdict_word is one of: PASS, FAIL_INCOMPLETE, FAIL_OFFTOPIC, FAIL_FORMAT.
    If no canonical verdict is detected, degrade gracefully to PASS.
    """
    text = (raw or "").strip()
    first = _safe_first_line(text, default="")

    # Prefer first-line canonical formats
    upper_first = first.upper()
    if upper_first == "PASS" or upper_first.startswith("PASS:"):
        return "PASS", first
    for cat in FAIL_CATEGORIES:
        if upper_first == cat or upper_first.startswith(cat + ":"):
            return cat, first

    # Fallback: scan full output for canonical tokens
    upper_all = text.upper()
    for cat in FAIL_CATEGORIES:
        if cat in upper_all:
            return cat, f"{cat}: inferred from grader output"
    if "PASS" in upper_all:
        return "PASS", "PASS"

    # Non-canonical grader output should be treated as formatting failure.
    return "FAIL_FORMAT", "FAIL_FORMAT: non-canonical grader output"


def _clean_code_output(query: str, text: str) -> str:
    """Normalize code-model output to reduce truncation/noise for code tasks.

    - Prefer fenced code contents when present.
    - If a recognizable code definition appears, trim leading prose before it.
    - If query asks for no docstring/commentary, strip comments/docstring lines.
    """
    if not text:
        return text

    cleaned = text.strip()

    # Prefer first fenced code block if present
    m = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()

    lines = cleaned.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(def|class)\s+", line) or re.match(r"^\s*(pub\s+)?fn\s+", line):
            start_idx = i
            break
    if start_idx is not None:
        cleaned = "\n".join(lines[start_idx:]).strip()

    ql = query.lower()
    wants_code_only = any(k in ql for k in [
        "just give me", "only code", "no commentary", "no comments", "no docstring",
    ])
    if not wants_code_only:
        return cleaned

    out_lines = []
    in_docstring = False
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        out_lines.append(line)
    cleaned = "\n".join(out_lines).strip()
    return cleaned


def _wants_code_only(query: str) -> bool:
    ql = query.lower()
    return any(k in ql for k in [
        "just give me", "only code", "no commentary", "no comments", "no docstring",
    ])


def _normalize_python_function_output(query: str, text: str) -> str:
    """Stable code normalizer used before display/grading for code tasks."""
    return _clean_code_output(query, text)


def _detect_code_language(query: str) -> str:
    q = (query or "").lower()
    if "rust" in q:
        return "rust"
    if "python" in q:
        return "python"
    if "javascript" in q or "js" in q:
        return "javascript"
    if "typescript" in q or "ts" in q:
        return "typescript"
    if "go" in q or "golang" in q:
        return "go"
    if "java" in q:
        return "java"
    if "c#" in q or "csharp" in q:
        return "csharp"
    if "c++" in q or "cpp" in q:
        return "cpp"
    return "generic"


def _describe_code_language(query: str) -> str:
    language = _detect_code_language(query)
    labels = {
        "rust": "Rust",
        "python": "Python",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "go": "Go",
        "java": "Java",
        "csharp": "C#",
        "cpp": "C++",
        "generic": "the requested programming language",
    }
    return labels.get(language, "the requested programming language")


def _grade_code_output(query: str, draft: str) -> Tuple[str, str]:
    """Deterministic grader for code tasks.

    This avoids brittle LLM self-grading for code responses and ensures
    architecture-level coherence: generation is model-based, validation is rule-based.
    """
    cleaned = _normalize_python_function_output(query, draft)
    if not cleaned:
        return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: empty output"

    language = _detect_code_language(query)

    if language == "rust":
        if not re.search(r"^\s*(?:pub\s+)?fn\s+[A-Za-z_]\w*\s*\(", cleaned, re.MULTILINE):
            return "FAIL_FORMAT", "FAIL_FORMAT: missing function definition"
        if cleaned.count("{") != cleaned.count("}"):
            return "FAIL_FORMAT", "FAIL_FORMAT: unbalanced braces"
        return "PASS", "PASS"

    if language == "python" or language == "generic":
        if not re.search(r"^\s*def\s+[A-Za-z_]\w*\s*\(", cleaned, re.MULTILINE):
            return "FAIL_FORMAT", "FAIL_FORMAT: missing function definition"
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            return "FAIL_FORMAT", f"FAIL_FORMAT: syntax error at line {e.lineno}"
        return "PASS", "PASS"

    generic_patterns = [
        r"^\s*(?:export\s+)?function\s+[A-Za-z_]\w*\s*\(",
        r"^\s*(?:public\s+|private\s+|static\s+)*(?:[A-Za-z_][\w<>\[\]]*\s+)+[A-Za-z_]\w*\s*\(",
        r"^\s*func\s+[A-Za-z_]\w*\s*\(",
    ]
    if not any(re.search(pattern, cleaned, re.MULTILINE) for pattern in generic_patterns):
        return "FAIL_FORMAT", "FAIL_FORMAT: missing function definition"

    return "PASS", "PASS"


def _is_html_web_request(query: str) -> bool:
    q = (query or "").lower()
    web_markers = ["html", "css", "javascript", "web app", "webapp", "self contained", "self-contained"]
    return sum(1 for marker in web_markers if marker in q) >= 2


def _grade_web_output(query: str, draft: str) -> Tuple[str, str]:
    text = (draft or "").strip().lower()
    if not text:
        return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: empty output"

    if _is_html_web_request(query):
        has_html = any(tok in text for tok in ["<!doctype html", "<html", "<head", "<body"])
        has_behavior = any(tok in text for tok in ["<script", "addEventListener", "setinterval", "settimeout", "function "])
        if not has_html:
            return "FAIL_FORMAT", "FAIL_FORMAT: missing HTML document structure"
        if not has_behavior:
            return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: missing timer behavior/javascript"

    return "PASS", "PASS"


def _deterministic_grade_guard(query: str, draft: str, domain: str) -> Optional[Tuple[str, str]]:
    """Return an obvious deterministic failure, else None.

    This is a confirmation guard only. It catches empty, scaffolded, or visibly
    malformed outputs before any model-based grading is trusted.
    """
    text = (draft or "").strip()
    if not text:
        return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: empty output"

    if domain == "code":
        word, line = _grade_code_output(query, draft)
        if word != "PASS":
            return word, line
        return None

    if domain == "web":
        web_word, web_line = _grade_web_output(query, draft)
        if web_word != "PASS":
            return web_word, web_line

    alpha_count = sum(ch.isalpha() for ch in text)
    punct_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    if len(text) >= 24 and alpha_count == 0 and punct_count >= max(12, len(text) // 2):
        return "FAIL_FORMAT", "FAIL_FORMAT: degenerate punctuation-only output"
    if len(text) >= 48 and punct_count > alpha_count * 3 and alpha_count < max(8, len(text) // 10):
        return "FAIL_FORMAT", "FAIL_FORMAT: highly degenerate symbol-heavy output"

    meta_prefixes = (
        "prompt for the ",
        "**prompt for the ",
        "prompt for ",
        "**prompt for ",
        "prompt\n",
        "prompt\r\n",
        'prompt\n"""',
        "medical specialist:",
        "for the medical specialist:",
        "specialist prompt:",
    )
    lower_text = text.lower()
    if lower_text.startswith(meta_prefixes):
        return "FAIL_FORMAT", "FAIL_FORMAT: prompt-echo output instead of an answer"
    if "output only the prompt text" in lower_text or "construct an optimal prompt" in lower_text:
        return "FAIL_FORMAT", "FAIL_FORMAT: meta-instructions leaked into output"
    if "context for code specialist" in lower_text or "insert missing context here" in lower_text:
        return "FAIL_FORMAT", "FAIL_FORMAT: leaked prompt scaffold instead of task output"

    return None


def _grade_with_specialist_self_check(
    query: str,
    draft: str,
    domain: str,
    mgr: "ModelManager",
) -> Tuple[str, str]:
    """Grade with the currently loaded specialist plus deterministic confirmation."""
    deterministic_fail = _deterministic_grade_guard(query, draft, domain)
    if deterministic_fail is not None:
        return deterministic_fail

    language_note = ""
    grader_role = f"strict {domain} specialist grader"
    if domain == "code":
        code_language = _describe_code_language(query)
        language_note = (
            f" The candidate should be evaluated as {code_language} code, not as Python unless the request explicitly asks for Python."
        )
        grader_role = f"strict {code_language} code grader"

    truncated_query = (query or "")[:600]
    truncated_draft = (draft or "")[:1600]
    grade_prompt = (
        f"User prompt:\n{truncated_query}\n\n"
        f"Candidate response:\n{truncated_draft}\n\n"
        "Evaluate whether the candidate response is accurate, complete, and on-topic. "
        f"Apply the grading rules for the requested domain carefully.{language_note} "
        "Output exactly one line starting with one of:\n"
        "PASS\n"
        "FAIL_INCOMPLETE: <brief reason>\n"
        "FAIL_OFFTOPIC: <brief reason>\n"
        "FAIL_FORMAT: <brief reason>"
    )
    verdict = mgr.generate_specialist(
        grade_prompt,
        system=(
            f"You are a {grader_role}. Judge the candidate response for the user's prompt."
            f"{language_note} "
            "Be skeptical. Do not rewrite the answer. Output only the verdict line."
        ),
        max_tokens=24,
        temperature=0.01,
    )
    verdict_word, verdict_line = _normalize_grade_verdict(verdict)
    deterministic_fail = _deterministic_grade_guard(query, draft, domain)
    if deterministic_fail is not None:
        return deterministic_fail
    if verdict_line == "FAIL_FORMAT: non-canonical grader output":
        return "PASS", "PASS: self-grader abstained; deterministic guards found no issue"
    return verdict_word, verdict_line


def _grade_output(
    query: str,
    draft: str,
    domain: str,
    mgr: "ModelManager",
    grade_system: str,
) -> Tuple[str, str]:
    """Policy-based grading router.

    - code: deterministic structural/syntax grading
    - others: LLM grader + canonical verdict normalization
    """
    deterministic_fail = _deterministic_grade_guard(query, draft, domain)
    if deterministic_fail is not None:
        return deterministic_fail

    if domain == "code":
        return "PASS", "PASS"

    language_note = ""
    if domain == "code":
        language_note = f" Evaluate code using {_describe_code_language(query)} conventions."

    grade_prompt = (
        f"Task: {query}\nOutput: {draft}\n\n"
        f"Assess the output. Reply with exactly one line, starting with one of:\n"
        f"PASS\n"
        f"FAIL_INCOMPLETE: <one line reason>\n"
        f"FAIL_OFFTOPIC: <one line reason>\n"
        f"FAIL_FORMAT: <one line reason>{language_note}"
    )
    verdict = mgr.generate_supervisor(
        grade_prompt,
        system=grade_system,
        max_tokens=24,
        temperature=0.01,
        think_budget=THINK_BUDGET_UTIL,
    )
    return _normalize_grade_verdict(verdict)


_VERDICT_QUALITY = {
    "PASS": 3,
    "FAIL_FORMAT": 2,
    "FAIL_INCOMPLETE": 1,
    "FAIL_OFFTOPIC": 0,
}


def _candidate_quality_score(verdict_word: str, text: str) -> float:
    """Return a comparable quality score for benchmark-style candidate selection."""
    base = _VERDICT_QUALITY.get((verdict_word or "").upper(), 0)
    length_bonus = min(len((text or "").strip()), 400) / 4000.0
    return base + length_bonus


def _pick_better_candidate(
    left_text: str,
    left_verdict: str,
    left_line: str,
    right_text: str,
    right_verdict: str,
    right_line: str,
) -> Tuple[str, str, str]:
    """Choose better response using verdict quality first, verbosity second."""
    left_score = _candidate_quality_score(left_verdict, left_text)
    right_score = _candidate_quality_score(right_verdict, right_text)

    if right_score > left_score:
        return right_text, right_verdict, right_line
    if right_score < left_score:
        return left_text, left_verdict, left_line

    if len((right_text or "").strip()) > len((left_text or "").strip()):
        return right_text, right_verdict, right_line
    return left_text, left_verdict, left_line


def _should_retry_tier2(verdict_word: str, text: str) -> bool:
    """Benchmark-style retry gate: retry once on weak/failed specialist draft."""
    if verdict_word in FAIL_CATEGORIES:
        return True
    return len((text or "").strip()) < 20


def _should_trigger_supervisor_fallback(verdict_word: str, domain: str) -> bool:
    """Benchmark-style fallback gate after primary pipeline selection."""
    if domain == "supervisor":
        return False
    return verdict_word in FAIL_CATEGORIES


def _run_supervisor_fallback(
    query: str,
    mgr: "ModelManager",
    current_answer: str,
    max_tokens: int,
    think_budget: int,
) -> str:
    """Generate a fallback answer directly from supervisor for failed outputs."""
    fb_prompt = (
        "Answer this question carefully and accurately.\n\n"
        f"QUESTION: {query}\n\n"
        f"Current candidate answer (may be flawed):\n{current_answer}\n\n"
        "Think through this step by step, then provide your final answer:"
    )
    return _generate_supervisor_answer_resilient(
        mgr=mgr,
        prompt=fb_prompt,
        system="You are an advanced reasoning supervisor.",
        max_tokens=max_tokens,
        temperature=0.2,
        think_budget=think_budget,
        retry_label="fallback",
    )


def _generate_supervisor_answer_resilient(
    mgr: "ModelManager",
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    think_budget: int,
    retry_label: str = "answer",
) -> str:
    """Generate with supervisor and retry once without thinking if empty."""
    result = mgr.generate_supervisor(
        prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        think_budget=think_budget,
    )
    if (result or "").strip():
        return result

    debug_log(f"supervisor.{retry_label}.empty_primary")
    retry_system = (
        f"{system}\n\n"
        "Answer directly and concretely. Do not spend time on hidden reasoning. "
        "If uncertain, say so briefly and then give the best grounded answer you can."
    ).strip()
    retry_result = mgr.generate_supervisor(
        prompt,
        system=retry_system,
        max_tokens=min(max_tokens, 1024),
        temperature=max(temperature, 0.1),
        think_budget=0,
    )
    if (retry_result or "").strip():
        debug_log(f"supervisor.{retry_label}.retry_recovered", output_chars=len(retry_result))
        return retry_result
    debug_log(f"supervisor.{retry_label}.retry_empty")
    return result


def _build_retry_guidance(
    query: str,
    task_prompt: str,
    fail_category: str,
    fail_reason: str,
    mgr: "ModelManager",
    kb_context: Optional[str] = None,
) -> str:
    """Generate concise supervisor guidance for retry without re-injecting failed output."""
    kb_block = f"\n\nReference material:\n{kb_context}" if kb_context else ""
    guidance_prompt = (
        f"Original user task: {query}\n\n"
        f"Current specialist task prompt:\n{task_prompt}\n\n"
        f"Failure category: {fail_category}\n"
        f"Failure reason: {fail_reason or 'unspecified'}"
        f"{kb_block}\n\n"
        "Produce retry guidance with exactly these 3 sections:\n"
        "1) What was wrong\n"
        "2) What to do now\n"
        "3) Output format requirements\n"
        "Keep it brief and actionable."
    )
    return mgr.generate_supervisor(
        guidance_prompt,
        system="You are a concise remediation planner for specialist retries.",
        max_tokens=220,
        temperature=0.1,
        think_budget=THINK_BUDGET_UTIL,
    )


def _format_prompt(model_path: str, prompt: str, system: str = "") -> str:
    """Format prompt per model architecture. Verbatim from run_coe_benchmark_v4.py."""
    p = model_path.lower()
    if "sqlcoder" in p:
        return (
            f"### Task\nGenerate a SQL query to answer the following question.\n\n"
            f"### Instructions\n{prompt}\n\n### Response\n"
        )
    if "law" in p:
        sys_part = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"
    if "biomistral" in p or "bio" in p:
        sys_part = f"{system}\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"
    # Qwen / DeepSeek / Nanbeige — ChatML format
    if system:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _estimate_token_count(text: str) -> int:
    """Cheap token estimate used when tokenizers live only inside workers."""
    s = text or ""
    if not s:
        return 0
    return max(1, (len(s) + 3) // 4)


def _trim_text_to_estimated_tokens(text: str, max_tokens: int) -> str:
    """Trim text conservatively using a rough chars-per-token estimate."""
    if max_tokens <= 0:
        return ""
    s = text or ""
    if _estimate_token_count(s) <= max_tokens:
        return s
    char_budget = max_tokens * 4
    return s[:char_budget]


@dataclass
class ModelHandle:
    path: str
    model: object       # og.Model
    tokenizer: object   # og.Tokenizer


class ModelManager:
    """
    Two-pool model lifecycle manager.

    Persistent pool (loaded at startup, never unloaded):
        _supervisor  — Nanbeige INT4 supervisor
        _embedder    — BGE-M3 embedding model (sentence-transformers)

    Swappable pool (per-task lifecycle, background load):
        _specialist  — one specialist at a time
    """

    def __init__(
        self,
        device: str = "dml",
        model_base_dir: Optional[str] = None,
        embed_cpu_batch_size: int = 96,
        embed_cpu_threads: int = 12,
        max_runtime_seq_tokens: int = 4096,
    ):
        self.device = device
        self._extra_dirs: List[str] = [model_base_dir] if model_base_dir else []
        self._embed_cpu_batch_size = max(8, int(embed_cpu_batch_size))
        self._embed_cpu_threads = max(1, int(embed_cpu_threads))
        self._max_runtime_seq_tokens = max(1024, int(max_runtime_seq_tokens))
        self._supervisor_path: Optional[str] = None
        self._embedder: Optional["EmbeddingManager"] = None
        self._specialist_path: Optional[str] = None
        self._load_thread: Optional[threading.Thread] = None
        self._pending_specialist_path: Optional[str] = None
        self._load_error: Optional[str] = None
        self._embedder_event: threading.Event = threading.Event()
        self._embed_thread: Optional[threading.Thread] = None

    def _resolve(self, name: str) -> str:
        """Resolve a bare model folder name to an absolute path."""
        return resolve_model_path(name, self._extra_dirs)

    def _load_supervisor_handle(self) -> None:
        sv_path = self._resolve(DOMAIN_MODELS["supervisor"])
        if not Path(sv_path).exists():
            raise FileNotFoundError(
                f"Supervisor model not found: {sv_path}\n"
                f"Search dirs: {[self._extra_dirs] + _DEFAULT_MODEL_SEARCH_DIRS}"
            )

        cprint(f"[ModelManager] Supervisor worker target: {sv_path}")
        t0 = time.time()
        self._supervisor_path = sv_path
        cprint(f"[ModelManager] Supervisor worker ready ({time.time() - t0:.1f}s)", "green")
        debug_log("startup.supervisor_ready", model=Path(sv_path).name, mode="worker")

    # ── Startup / Shutdown ────────────────────────────────────────────────

    def startup(self) -> None:
        """Load supervisor synchronously; start embedder load in a background thread."""
        debug_log("startup.begin", device=self.device)

        self._load_supervisor_handle()

        if EMBED_AVAILABLE:
            # dml is an OGA/ONNX concept; sentence-transformers uses PyTorch/ROCm.
            # To avoid VRAM contention/creep with OGA models, default embedder to CPU
            # for dml workflows. Keep GPU embedder only for explicit "cuda" runs.
            embed_device = "cuda" if self.device == "cuda" else "cpu"
            cprint(f"[ModelManager] Loading embedder: {EMBEDDING_MODEL} on {embed_device} (background…)")
            self._embed_thread = threading.Thread(
                target=self._load_embedder_bg, args=(embed_device,), daemon=True
            )
            self._embed_thread.start()
        else:
            cprint("[ModelManager] sentence-transformers not available — embedding features disabled", "yellow")
            debug_log("startup.embedder_unavailable")
            self._embedder_event.set()  # nothing to wait for

    def _load_embedder_bg(self, embed_device: str) -> None:
        """Background thread: load EmbeddingManager and set the ready event."""
        t0 = time.time()
        try:
            self._embedder = EmbeddingManager(
                model_name=EMBEDDING_MODEL,
                device=embed_device,
                cpu_batch_size=self._embed_cpu_batch_size,
                cpu_threads=self._embed_cpu_threads,
            )
            cprint(f"[ModelManager] Embedder ready ({time.time() - t0:.1f}s)", "green")
            debug_log("startup.embedder_ready", device=embed_device)
        except Exception as e:
            cprint(f"[ModelManager] Embedder load failed: {e}", "yellow")
        finally:
            self._embedder_event.set()

    def await_embedder(self, verbose: bool = True) -> Optional["EmbeddingManager"]:
        """Block until the background embedder thread has finished. Returns the embedder (or None)."""
        if not self._embedder_event.is_set():
            if verbose:
                cprint("[CoE] Embedder still loading — please wait a moment…", "dim")
            self._embedder_event.wait()
        return self._embedder

    def shutdown(self) -> None:
        """Unload all models."""
        debug_log("shutdown.begin")
        if self._load_thread and self._load_thread.is_alive():
            self._load_thread.join()
        self.unload_specialist()
        self._supervisor_path = None
        self._embedder = None
        gc.collect()
        cprint("[ModelManager] Shutdown complete.", "dim")
        debug_log("shutdown.complete")

    def reset_to_supervisor_only(self, reason: str = "") -> None:
        """Return to a neutral state with supervisor kept resident.

        DirectML/OGA appears unstable when unloading a specialist and then
        immediately tearing down/recreating the supervisor in the same process.
        Keep the supervisor alive, unload only the specialist, and compact
        CPU-side helpers. A true startup-identical VRAM state would require a
        full process restart.
        """
        debug_log("reset.begin", reason=reason or "manual")
        if self._load_thread and self._load_thread.is_alive():
            self._load_thread.join()
        self.unload_specialist()
        self._pending_specialist_path = None
        self._load_error = None

        if self._supervisor_path:
            debug_log(
                "reset.supervisor_kept",
                model=Path(self._supervisor_path).name,
                mode="worker",
                reason=reason or "manual",
            )

        if self._embedder and hasattr(self._embedder, "compact"):
            try:
                self._embedder.compact()
            except Exception:
                pass
        gc.collect()
        debug_log("reset.end", reason=reason or "manual")

    # ── Specialist lifecycle ──────────────────────────────────────────────

    def ensure_specialist_for_model(self, model_path: str) -> str:
        """Apply domain-sticky specialist policy for TIER2/TIER3.

        - Reuse currently loaded specialist when model_path matches
        - Force unload when model_path changes
        - Start background load for the requested model when needed

        Returns: "reuse" | "switch" | "load"
        """
        if self._load_thread and self._load_thread.is_alive():
            self._load_thread.join()

        current = self._specialist_path
        if current == model_path:
            return "reuse"

        if current and current != model_path:
            cprint(
                f"[ModelManager] Specialist switch: {Path(current).name} -> {Path(model_path).name}",
                "dim",
            )
            self.unload_specialist()
            self.begin_load_specialist(model_path)
            return "switch"

        self.begin_load_specialist(model_path)
        return "load"

    def begin_load_specialist(self, model_path: str) -> None:
        """Register specialist target path for worker-based inference."""
        if self._specialist_path == model_path:
            return
        self._load_error = None
        self._pending_specialist_path = model_path
        self._do_load_specialist(model_path)

    def _do_load_specialist(self, model_path: str) -> None:
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Specialist model not found: {model_path}")
            t0 = time.time()
            self._specialist_path = model_path
            cprint(
                f"[ModelManager] Specialist ready: {Path(model_path).name} "
                f"({time.time() - t0:.1f}s) mode=worker",
                "green",
            )
            debug_log("specialist.load_ready", model=Path(model_path).name, mode="worker")
        except Exception as e:
            self._load_error = str(e)
            cprint(f"[ModelManager] Specialist load FAILED: {e}", "red")
            debug_log("specialist.load_failed", model=Path(model_path).name, error=str(e))
        finally:
            self._pending_specialist_path = None

    def await_specialist(self) -> None:
        """Compatibility no-op for worker-based specialist routing."""
        return None

    def unload_specialist(self) -> None:
        if self._specialist_path:
            old_model_name = Path(self._specialist_path).name
            self._specialist_path = None
            gc.collect()
            cprint(f"[ModelManager] Specialist unloaded. mode=worker")
            debug_log("specialist.unloaded", model=old_model_name, mode="worker")

    def specialist_path(self) -> Optional[str]:
        return self._specialist_path

    def reconcile_residency(self, expected_domain: Optional[str], reason: str = "") -> None:
        """Post-inference residency audit/reconciliation.

        Keeps current specialist resident by default. If an expected domain is
        provided (TIER2/TIER3), ensures the matching specialist is the only
        active specialist model.
        """
        self.await_specialist()

        current = self.specialist_path()
        if current and not Path(current).exists():
            cprint(f"[Residency] Loaded specialist path missing on disk: {current} — unloading", "yellow")
            self.unload_specialist()
            current = None

        expected_path = None
        if (
            expected_domain
            and expected_domain in DOMAIN_MODELS
            and expected_domain != "supervisor"
            and expected_domain not in DISABLED_SPECIALIST_DOMAINS
        ):
            expected_path = self._resolve(DOMAIN_MODELS[expected_domain])

        if expected_path and current != expected_path:
            action = self.ensure_specialist_for_model(expected_path)
            self.await_specialist()
            if self._load_error:
                cprint(f"[Residency] reconcile failed ({reason}): {self._load_error}", "yellow")
            else:
                cprint(f"[Residency] reconcile {reason}: {action} -> {Path(expected_path).name}", "dim")
        elif expected_path and current == expected_path:
            cprint(f"[Residency] reconcile {reason}: keep {Path(expected_path).name}", "dim")
        elif current:
            cprint(f"[Residency] reconcile {reason}: keep current specialist {Path(current).name}", "dim")
        else:
            cprint(f"[Residency] reconcile {reason}: no specialist resident", "dim")

        if self._embedder and hasattr(self._embedder, "compact"):
            try:
                self._embedder.compact()
            except Exception:
                pass

    def supervisor_path(self) -> str:
        return self._supervisor_path or ""

    def _generate_via_worker(
        self,
        model_path: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        think_budget: int = THINK_BUDGET_T2,
    ) -> Tuple[str, float]:
        if not INFERENCE_WORKER_PATH.exists():
            raise RuntimeError(f"Inference worker not found: {INFERENCE_WORKER_PATH}")

        request = {
            "model_path": model_path,
            "prompt": prompt,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "think_budget": think_budget,
            "max_runtime_seq_tokens": self._max_runtime_seq_tokens,
        }

        with tempfile.TemporaryDirectory(prefix="coe_worker_") as td:
            req_path = Path(td) / "request.json"
            resp_path = Path(td) / "response.json"
            req_path.write_text(json.dumps(request), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(INFERENCE_WORKER_PATH), str(req_path), str(resp_path)],
                cwd=str(_HERE),
                capture_output=True,
                text=True,
            )

            payload = None
            if resp_path.exists():
                payload = json.loads(resp_path.read_text(encoding="utf-8"))

            if not payload or not payload.get("ok"):
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                detail = ""
                if payload and payload.get("error"):
                    detail = payload["error"]
                elif stderr:
                    detail = stderr.splitlines()[-1]
                elif stdout:
                    detail = stdout.splitlines()[-1]
                elif completed.returncode != 0:
                    detail = f"worker exited with code {completed.returncode}"
                else:
                    detail = "worker produced no response"
                raise RuntimeError(f"Worker inference failed for {Path(model_path).name}: {detail}")

            return str(payload.get("result", "")), float(payload.get("elapsed_s", 0.0))

    # ── Generation kernel ─────────────────────────────────────────────────

    def _generate_with(
        self,
        handle: ModelHandle,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        think_budget: int = THINK_BUDGET_T2,
    ) -> str:
        import onnxruntime_genai as og

        formatted = _format_prompt(handle.path, prompt, system)
        input_tokens = handle.tokenizer.encode(formatted)

        # Respect model context limit
        ctx_limit = 4096
        cfg_file = Path(handle.path) / "genai_config.json"
        if cfg_file.exists():
            try:
                # utf-8-sig strips the BOM present in some model config files
                cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
                ctx_limit = cfg.get("model", {}).get("context_length", 4096)
            except Exception:
                pass

        temp = max(temperature, 0.01)

        effective_ctx_limit = min(ctx_limit, self._max_runtime_seq_tokens)
        if len(input_tokens) > effective_ctx_limit - 32:
            input_tokens = input_tokens[-(effective_ctx_limit - 32):]
            cprint(
                f"[GenGuard] prompt clipped to {len(input_tokens)} tokens "
                f"(cap={effective_ctx_limit}, model={Path(handle.path).name})",
                "dim",
            )

        if _is_thinking_model(handle.path):
            # Hard safety invariant: never create a generator when the
            # formatted input already fills (or exceeds) model context.
            initial_headroom = effective_ctx_limit - len(input_tokens) - 1
            if initial_headroom <= 0:
                return ""

            # ── Phase 1: thinking ────────────────────────────────────────
            # Run up to think_budget tokens.  Stop early when </think> is
            # seen in the decoded output so the budget is not wasted.
            reserve_for_answer = min(max_tokens, max(256, min(1024, initial_headroom // 3)))
            think_limit = min(think_budget, max(initial_headroom - reserve_for_answer - 1, 0))
            # Skip phase-1 safely when no room is available.
            if think_limit < 1:
                think_tokens = []
                found_close = True
                think_text = ""
            else:
                p1 = og.GeneratorParams(handle.model)
                p1.set_search_options(
                    max_length=min(len(input_tokens) + think_limit, effective_ctx_limit),
                    temperature=temp,
                )
                g1 = og.Generator(handle.model, p1)
                g1.append_tokens(input_tokens)

                think_tokens: list = []
                found_close = False
                while not g1.is_done() and len(think_tokens) < think_limit:
                    g1.generate_next_token()
                    think_tokens.append(g1.get_next_tokens()[0])
                    # Check decoded text every 4 tokens (cheap, avoids per-token decode)
                    if len(think_tokens) % 4 == 0:
                        if "</think>" in handle.tokenizer.decode(think_tokens).lower():
                            found_close = True
                            break

                think_text = handle.tokenizer.decode(think_tokens)
                del g1
                del p1

            # If the model never closed the block, force-close it so Phase 2
            # sees a complete <think>...</think> and knows to write the answer.
            if not found_close:
                think_text = think_text.rstrip() + "\n</think>\n"

            # ── Phase 2: answer ──────────────────────────────────────────
            # Feed (formatted_prompt + complete think block) as the new
            # context; the model now only needs to produce the answer.
            phase2_text = formatted + think_text
            phase2_tokens = handle.tokenizer.encode(phase2_text)
            if len(phase2_tokens) > effective_ctx_limit - 32:
                phase2_tokens = phase2_tokens[-(effective_ctx_limit - 32):]
            answer_budget = min(max_tokens, effective_ctx_limit - len(phase2_tokens) - 10)
            if answer_budget <= 0:
                return ""

            p2 = og.GeneratorParams(handle.model)
            p2.set_search_options(
                max_length=min(len(phase2_tokens) + answer_budget, effective_ctx_limit),
                temperature=temp,
            )
            g2 = og.Generator(handle.model, p2)
            g2.append_tokens(phase2_tokens)

            answer_tokens: list = []
            while not g2.is_done() and len(answer_tokens) < answer_budget:
                g2.generate_next_token()
                answer_tokens.append(g2.get_next_tokens()[0])

            response = handle.tokenizer.decode(answer_tokens)
            del g2
            del p2
            # Phase 2 output should be pure answer, but strip any stray think
            # blocks just in case (e.g. model re-opens one).
            return _strip_think(response).strip()

        else:
            # ── Non-thinking model: single phase ─────────────────────────
            available_gen = min(max_tokens, effective_ctx_limit - len(input_tokens) - 10)
            if available_gen <= 0:
                return ""

            params = og.GeneratorParams(handle.model)
            params.set_search_options(
                max_length=min(len(input_tokens) + available_gen, effective_ctx_limit),
                temperature=temp,
            )
            generator = og.Generator(handle.model, params)
            generator.append_tokens(input_tokens)

            output_tokens: list = []
            while not generator.is_done() and len(output_tokens) < available_gen:
                generator.generate_next_token()
                output_tokens.append(generator.get_next_tokens()[0])

            response = handle.tokenizer.decode(output_tokens)
            del generator
            del params
            return response.strip()

    def generate_supervisor(
        self, prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.2,
        think_budget: int = THINK_BUDGET_T2,
    ) -> str:
        if not self._supervisor_path:
            raise RuntimeError("Supervisor not loaded — call startup() first")
        debug_log(
            "gen.supervisor.begin",
            model=Path(self._supervisor_path).name,
            prompt_chars=len(prompt or ""),
            max_tokens=max_tokens,
            think_budget=think_budget,
            temperature=temperature,
        )
        result, elapsed_s = self._generate_via_worker(
            self._supervisor_path,
            prompt,
            system,
            max_tokens,
            temperature,
            think_budget,
        )
        debug_log(
            "gen.supervisor.end",
            model=Path(self._supervisor_path).name,
            elapsed_s=round(elapsed_s, 3),
            output_chars=len(result or ""),
        )
        return result

    def generate_specialist(
        self, prompt: str, system: str = "", max_tokens: int = OUTPUT_BUDGET_T2,
        temperature: float = 0.3,
    ) -> str:
        # Specialist models are never thinking models; _generate_with skips
        # two-phase logic automatically, so no think_budget is needed here.
        self.await_specialist()
        if not self._specialist_path:
            raise RuntimeError(f"Specialist not loaded. Error: {self._load_error}")
        debug_log(
            "gen.specialist.begin",
            model=Path(self._specialist_path).name,
            prompt_chars=len(prompt or ""),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result, elapsed_s = self._generate_via_worker(
            self._specialist_path,
            prompt,
            system,
            max_tokens,
            temperature,
            THINK_BUDGET_T2,
        )
        debug_log(
            "gen.specialist.end",
            model=Path(self._specialist_path).name,
            elapsed_s=round(elapsed_s, 3),
            output_chars=len(result or ""),
        )
        return result

    def supervisor_ctx_limit(self) -> int:
        """Return the supervisor's context window size in tokens.
        Read directly from genai_config.json so it is always correct
        regardless of which supervisor model is loaded."""
        if not self._supervisor_path:
            return 4096
        cfg_file = Path(self._supervisor_path) / "genai_config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
                return cfg.get("model", {}).get("context_length", 4096)
            except Exception:
                pass
        return 4096


def _fit_draft(draft: str, prompt_skeleton: str, mgr: "ModelManager",
               output_budget: int, think_budget: int, system: str = "",
               max_draft_tokens: Optional[int] = None) -> str:
    """Trim *draft* to the largest token-exact slice that still leaves room
    for the prompt skeleton, think phase, and answer budget within the
    supervisor's actual context window.  Replaces any hard-coded char limit."""
    if not draft:
        return draft
    if not mgr.supervisor_path():
        return draft
    ctx = mgr.supervisor_ctx_limit()
    formatted = _format_prompt(mgr.supervisor_path(), prompt_skeleton, system)
    skeleton_toks = _estimate_token_count(formatted)
    # Reserve space: output tokens + think budget + small safety margin
    reserve = output_budget + think_budget + 32
    available = max(ctx - skeleton_toks - reserve, 0)
    if max_draft_tokens is not None:
        available = min(available, max_draft_tokens)
    if available == 0:
        return ""  # nothing fits; supervisor works from query alone
    if _estimate_token_count(draft) <= available:
        return draft
    return _trim_text_to_estimated_tokens(draft, available)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Two-Stage Tier Classifier
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_KEYWORDS: Dict[str, set] = {
    "code":    {"python", "function", "class", "algorithm", "implement", "code", "script",
                "debug", "refactor", "unit test", "async", "loop", "decorator", "module"},
    "sql":     {"sql", "query", "database", "table", "schema", "join", "select", "insert",
                "index", "postgresql", "mysql", "sqlite"},
    "math":    {"math", "calculus", "equation", "matrix", "probability", "integral",
                "derivative", "statistics", "proof", "theorem", "algebra"},
    "medical": {"medical", "clinical", "patient", "diagnosis", "treatment", "drug",
                "symptom", "disease", "health", "anatomy", "pharmacology"},
    "legal":   {"legal", "contract", "clause", "liability", "statute", "regulation",
                "court", "plaintiff", "defendant", "jurisdiction", "compliance",
                "law", "murder", "theft", "robbery", "burglary", "assault",
                "homicide", "felony", "misdemeanor", "crime", "criminal",
                "statute of limitations", "first degree", "second degree"},
    "web":     {"html", "css", "javascript", "react", "vue", "frontend", "backend", "api",
                "rest", "typescript", "component", "dom", "responsive", "timer", "button",
                "theme", "dark theme", "ui"},
}

_TRIVIAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|help|what can you do|thanks|thank you|bye|quit|exit)\??\.?$",
    re.IGNORECASE,
)


def classify_query(
    query: str,
    mgr: "ModelManager",
    confirm_enabled: bool = True,
) -> Tuple[str, List[str], str]:
    """
    Two-stage classifier.
    Returns (tier, domains, interpretation).
    Never returns empty domains — fallback is ("TIER2", ["supervisor"], "General query").
    """
    valid_domains = set(DOMAIN_MODELS.keys())
    ql = query.lower().strip()
    words = query.split()

    def _is_general_factual_query(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t or len(t.split()) > 14:
            return False
        factual_starts = (
            "what is ",
            "what's ",
            "who is ",
            "when is ",
            "when was ",
            "where is ",
            "how fast ",
            "how far ",
            "how many ",
            "how much ",
            "what does ",
            "define ",
        )
        if not t.endswith("?") and not any(t.startswith(p) for p in factual_starts):
            return False
        if not any(t.startswith(p) for p in factual_starts):
            return False
        disqualifiers = (
            "write ",
            "build ",
            "create ",
            "implement ",
            "design ",
            "story",
            "poem",
            "html",
            "sql",
            "function",
            "code",
            "api",
            "contract",
            "diagnosis",
        )
        return not any(marker in t for marker in disqualifiers)

    def _normalize_domains(raw_domains: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for item in raw_domains or []:
            dom = str(item).strip().lower()
            if dom in valid_domains and dom not in seen:
                normalized.append(dom)
                seen.add(dom)
        return normalized

    def _enforce_routing_contract(tier_in: str, domains_in: List[str]) -> Tuple[str, List[str]]:
        tier_norm = (tier_in or "TIER2").upper()
        doms_norm = _normalize_domains(domains_in)
        if not doms_norm:
            doms_norm = ["supervisor"]

        # Hard guard: TIER1 is only for truly trivial supervisor-only interactions.
        # Any domain-directed or non-trivial request must route to TIER2/TIER3.
        if tier_norm == "TIER1":
            if doms_norm != ["supervisor"] or not (_TRIVIAL_PATTERNS.match(ql) or _is_general_factual_query(ql)):
                if len(doms_norm) >= 2:
                    return "TIER3", doms_norm
                return "TIER2", doms_norm

        if tier_norm != "TIER1" and len(doms_norm) >= 2:
            return "TIER3", doms_norm
        if tier_norm == "TIER3" and len(doms_norm) < 2:
            return "TIER2", doms_norm
        if tier_norm not in ("TIER1", "TIER2", "TIER3"):
            return "TIER2", doms_norm
        return tier_norm, doms_norm

    def _is_veterinary_query(text: str) -> bool:
        animal_terms = {
            "cat", "cats", "dog", "dogs", "kitten", "kittens", "puppy", "puppies",
            "pet", "pets", "feline", "canine", "rabbit", "hamster", "bird", "parrot",
            "horse", "cow", "goat", "sheep", "veterinary", "vet",
        }
        health_terms = {
            "temperature", "fever", "symptom", "symptoms", "normal", "body temperature",
            "dose", "dosage", "medicine", "medication", "illness", "disease", "vomiting",
            "diarrhea", "injury", "pain", "heart rate", "breathing", "health",
        }
        has_animal = any(_keyword_match(term, text) for term in animal_terms)
        has_health = any(_keyword_match(term, text) for term in health_terms)
        return has_animal and has_health

    def _is_single_artifact_web_task(text: str) -> bool:
        t = (text or "").lower()
        has_web = any(k in t for k in ["html", "css", "javascript", "web app", "webapp", "timer", "button"])
        asks_single = any(k in t for k in ["self contained", "self-contained", "complete", "single html", "one file"])
        return has_web and asks_single

    def _is_creative_writing_query(text: str) -> bool:
        t = (text or "").lower().strip()
        creative_markers = [
            "write a short story",
            "write me a story",
            "tell me a story",
            "short story",
            "fiction",
            "poem",
            "haiku",
            "sonnet",
            "creative writing",
            "write a narrative",
        ]
        return any(marker in t for marker in creative_markers)

    def _keyword_match(keyword: str, text: str) -> bool:
        """Match keywords by token boundary where possible to avoid substring collisions.
        Example: "functional" should not match keyword "function".
        """
        kw = keyword.strip().lower()
        if not kw:
            return False
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(kw) + r'(?![a-zA-Z0-9])'
        return bool(re.search(pattern, text))

    # ── Stage 1: Heuristic scan ──────────────────────────────────────────
    # True greetings / meta-commands → TIER1 immediately, no supervisor call
    if _TRIVIAL_PATTERNS.match(ql):
        return "TIER1", ["supervisor"], query[:80]

    # Short general factual/science/reference questions should route directly
    # to supervisor TIER1 instead of entering stage-2 and drifting into absurd
    # multi-domain classifications.
    if _is_general_factual_query(ql):
        return "TIER1", ["supervisor"], f"A general factual question: {query[:80]}"

    # Veterinary health questions are safer on the supervisor path than on
    # the human-clinical specialist, and they should never escalate into a
    # spurious multi-domain route.
    if _is_veterinary_query(ql):
        return "TIER2", ["supervisor"], f"A veterinary health question: {query[:80]}"

    # Open-ended creative writing is better handled by the supervisor than by
    # any domain specialist, and it should not drift into web/code routing.
    if _is_creative_writing_query(ql):
        return "TIER2", ["supervisor"], f"A creative writing request: {query[:80]}"

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if _keyword_match(kw, ql))
        if count:
            scores[domain] = count

    sorted_domains = sorted(scores, key=lambda d: scores[d], reverse=True)

    is_trivial = len(words) < 4  # short queries: skip high-confidence shortcut, go Stage 2
    is_multi = (
        len(sorted_domains) >= 2
        and len(words) > 40
    )

    # Determine confidence
    if len(sorted_domains) >= 2:
        top, second = scores[sorted_domains[0]], scores[sorted_domains[1]]
        high_confidence = top > 2 * second and not is_multi
    elif len(sorted_domains) == 1:
        high_confidence = not is_trivial and not is_multi
    else:
        high_confidence = False

    # TIER1 or TIER3 candidates → always go Stage 2
    if is_trivial:
        pass  # Stage 2 will classify short-but-meaningful queries (e.g. "write quicksort")
    elif is_multi:
        pass  # Stage 2 below
    elif high_confidence and sorted_domains:
        # TIER2 high confidence — skip Stage 2
        interpretation = f"A {sorted_domains[0]} task: {query[:80]}"
        return "TIER2", [sorted_domains[0]], interpretation

    # ── Stage 2: Supervisor classification ──────────────────────────────
    prompt = (
        "Classify this query for routing. Output JSON with keys:\n"
        "  tier: \"TIER1\" | \"TIER2\" | \"TIER3\"\n"
        "  domains: [list of domain keys from: code, sql, math, medical, legal, web, supervisor]\n"
        "  interpretation: one-sentence summary of what is being asked\n"
        "  confidence: \"high\" | \"low\"\n"
        "  recheck_needed: true | false\n\n"
        f"Query: {query}"
    )
    system = "You are a precise routing classifier. Output only valid JSON."

    def _run_stage2(qtext: str) -> Tuple[str, List[str], str]:
        try:
            raw = mgr.generate_supervisor(qtext, system=system, max_tokens=120, temperature=0.01, think_budget=THINK_BUDGET_UTIL)
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
            # Find JSON object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON found")
            parsed = json.loads(m.group(0))
            tier = str(parsed.get("tier", "TIER2")).upper()
            doms = [str(d).lower() for d in parsed.get("domains", [])]
            interp = str(parsed.get("interpretation", query[:80]))
            recheck = parsed.get("recheck_needed", False)
            tier, doms = _enforce_routing_contract(tier, doms)
            return tier, doms, interp, recheck
        except Exception as e:
            cprint(f"[Classifier] Stage 2 parse error: {e}", "yellow")
            return "TIER2", ["supervisor"], query[:80], False

    tier, doms, interp, recheck = _run_stage2(prompt)

    # One recheck if supervisor flagged it
    if recheck:
        recheck_prompt = (
            f"Re-classify with more precision. Previous domains: {doms}.\n\n{prompt}"
        )
        tier, doms, interp, _ = _run_stage2(recheck_prompt)

    tier, doms = _enforce_routing_contract(tier, doms)

    # Guard: if stage-2 overclassifies a short factual question into a large
    # multi-domain set, collapse it back to direct supervisor handling.
    if _is_general_factual_query(ql) and (len(doms) >= 2 or set(doms) == valid_domains):
        return "TIER1", ["supervisor"], f"A general factual question: {query[:80]}"

    # Guard: single-artifact HTML web tasks should not split into code+web TIER3.
    if tier == "TIER3" and set(doms) == {"code", "web"} and _is_single_artifact_web_task(query):
        return "TIER2", ["web"], interp

    # BioMistral is tuned for human clinical text. Veterinary questions are
    # safer routed through the supervisor-only path unless a dedicated animal
    # health specialist exists.
    if "medical" in doms and _is_veterinary_query(ql):
        return "TIER2", ["supervisor"], f"A veterinary health question: {query[:80]}"

    return tier, doms, interp


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Qualifier Infrastructure & Pipelines
# ═══════════════════════════════════════════════════════════════════════════

QUALIFIER_KEYWORDS: Dict[str, Dict[str, str]] = {
    "language": {
        "python": "Python", "c++": "C++", "cpp": "C++", "c#": "C#", "csharp": "C#",
        "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript",
        "go": "Go", "golang": "Go", "rust": "Rust", "swift": "Swift",
        "kotlin": "Kotlin", "ruby": "Ruby", "php": "PHP",
    },
    "web_framework": {
        "react": "React", "vue": "Vue", "angular": "Angular", "svelte": "Svelte",
        "next.js": "Next.js", "nuxt": "Nuxt", "htmx": "HTMX",
    },
    "sql_dialect": {
        "postgresql": "PostgreSQL", "postgres": "PostgreSQL", "mysql": "MySQL",
        "sqlite": "SQLite", "mssql": "SQL Server", "bigquery": "BigQuery",
        "snowflake": "Snowflake",
    },
    "subspecialty": {
        "cardiology": "cardiology", "oncology": "oncology", "pediatrics": "pediatrics",
        "pharmacology": "pharmacology", "neurology": "neurology",
        "gdpr": "GDPR/EU compliance", "hipaa": "HIPAA/US healthcare law",
        "uk law": "UK law", "eu law": "EU law", "ip law": "IP law",
        "contract law": "contract law",
    },
    "style": {
        "with unit tests": "with unit tests", "with tests": "with unit tests",
        "with docstrings": "with docstrings", "production-grade": "production-grade",
        "production grade": "production-grade", "functional": "functional style",
        "beginner-friendly": "beginner-friendly", "beginner": "beginner-friendly",
        "concise": "concise", "step-by-step": "step-by-step",
        "with comments": "with inline comments",
    },
    "level": {
        "advanced": "advanced", "intermediate": "intermediate",
        "expert": "expert-level", "academic": "academic/formal",
    },
}


@dataclass
class QueryQualifiers:
    language:     Optional[str] = None
    framework:    Optional[str] = None
    dialect:      Optional[str] = None
    subspecialty: Optional[str] = None
    style:        Optional[str] = None
    level:        Optional[str] = None

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in [
            ("language", self.language), ("framework", self.framework),
            ("dialect", self.dialect), ("subspecialty", self.subspecialty),
            ("style", self.style), ("level", self.level),
        ] if v]
        return ", ".join(parts) if parts else "none"

    def has_any(self) -> bool:
        return any([self.language, self.framework, self.dialect,
                    self.subspecialty, self.style, self.level])


def extract_qualifiers(query: str) -> QueryQualifiers:
    """Rule-based qualifier extraction. Zero model cost. Case-insensitive.
    Uses alphanumeric boundaries (?<![a-zA-Z0-9]) so C++, C#, Next.js etc. match correctly."""
    ql = query.lower()
    q = QueryQualifiers()

    def _word_match(kw: str, text: str) -> bool:
        """Match keyword not adjacent to alphanumeric characters."""
        return bool(re.search(
            r'(?<![a-zA-Z0-9])' + re.escape(kw) + r'(?![a-zA-Z0-9])',
            text
        ))

    for kw, val in QUALIFIER_KEYWORDS["language"].items():
        if _word_match(kw, ql):
            q.language = val; break
    for kw, val in QUALIFIER_KEYWORDS["web_framework"].items():
        if kw in ql: q.framework = val; break
    for kw, val in QUALIFIER_KEYWORDS["sql_dialect"].items():
        if kw in ql: q.dialect = val; break
    for kw, val in QUALIFIER_KEYWORDS["subspecialty"].items():
        if kw in ql: q.subspecialty = val; break
    for kw, val in QUALIFIER_KEYWORDS["style"].items():
        if kw in ql: q.style = val; break
    for kw, val in QUALIFIER_KEYWORDS["level"].items():
        if _word_match(kw, ql):
            q.level = val; break
    return q


SPECIALIST_PERSONA_TEMPLATES: Dict[str, str] = {
    "code":       "You are an expert {language} engineer specializing in clean, efficient, well-documented code.{style_clause}",
    "sql":        "You are an expert {dialect} database architect specializing in query optimization and schema design.",
    "math":       "You are an expert mathematician providing rigorous, step-by-step solutions.{level_clause}",
    "medical":    "You are an expert clinician{subspecialty_clause} providing evidence-based clinical information.",
    "legal":      "You are an expert legal analyst specializing in {subspecialty_or_default}.{level_clause}",
    "web":        "You are an expert {framework} developer specializing in modern web applications and REST APIs.{style_clause}",
    "supervisor": "You are a knowledgeable generalist assistant.",
}


def patch_persona(domain: str, q: QueryQualifiers) -> str:
    """Instantiate specialist persona template with detected qualifiers."""
    t = SPECIALIST_PERSONA_TEMPLATES.get(domain, SPECIALIST_PERSONA_TEMPLATES["supervisor"])
    return t.format(
        language=q.language or "software",
        dialect=q.dialect or "relational",
        framework=(q.framework or "full-stack web") + " and REST API",
        subspecialty_clause=f" specializing in {q.subspecialty}" if q.subspecialty else "",
        subspecialty_or_default=q.subspecialty or "contract law and regulatory compliance",
        style_clause=f" Focus on {q.style}." if q.style else "",
        level_clause=f" Tailor for {q.level} audience." if q.level else "",
    )


def build_supervisor_answer_system(base_system: str, domain: str) -> str:
    """Add a domain-grounded honesty preface for supervisor answer generation."""
    domain_label = (domain or "general topics").replace("_", " ")
    preface = (
        f"You are an expert in {domain_label} and able to give grounded, well-thought-out "
        f"information about {domain_label}. However, you are also honest and willing to tell "
        "the user if you are not sure of an answer or if there are substantial and serious "
        "conflicting schools of thought about a particular subject."
    )
    base = (base_system or "").strip()
    if not base:
        return preface
    return f"{preface}\n\n{base}"


def build_supervisor_tier2_specialist_system(base_system: str, domain: str) -> str:
    """Stricter system prompt for supervisor acting in TIER2 specialist capacity."""
    domain_label = (domain or "general topics").replace("_", " ")
    shared_rules = (
        f"You are acting as the {domain_label} specialist for this request. "
        "Be precise, grounded, and conservative. Prefer a shorter accurate answer over a longer speculative one. "
        "Do not invent citations, statutes, case names, technical terms, or factual details. "
        "If a detail is uncertain, say that it varies by jurisdiction, source, version, or context rather than guessing. "
        "Do not pad the answer with broad comparisons unless they are necessary to answer the user's actual question. "
        "Do not include meta-commentary about your process."
    )

    domain_rules = {
        "legal": (
            "For legal topics, do not present legal information as universal when it is jurisdiction-dependent. "
            "Explicitly say when a concept varies by jurisdiction. "
            "If the user did not specify a jurisdiction, give only a clearly labeled general overview and note the limitation. "
            "Do not cite statutes, section numbers, regulations, cases, or named legal tests unless the user asked for a specific jurisdiction and you are confident they are real and relevant. "
            "Do not provide a multi-country or multi-state survey unless the user explicitly asks for comparison. "
            "Avoid edge-case exceptions, sentencing details, historical background, and illustrative cases unless they are necessary to answer the user's actual question. "
            "Prefer a short high-level definition with a brief note that details vary by jurisdiction."
        ),
        "medical": (
            "For medical topics, distinguish clearly between general educational information and individualized medical advice. "
            "Do not fabricate study findings, dosages, contraindications, or diagnostic certainty."
        ),
        "code": (
            "For code tasks, satisfy the requested language and output format exactly. "
            "Do not switch languages or add explanatory prose when code-only output is requested."
        ),
    }

    base = build_supervisor_answer_system(base_system, domain)
    extra = domain_rules.get(domain, "")
    return f"{base}\n\n{shared_rules}" + (f"\n\n{extra}" if extra else "")


# ── TIER1 Pipeline ────────────────────────────────────────────────────────

def run_tier1(query: str, mgr: ModelManager) -> str:
    """TIER1: supervisor answers directly. No specialist load/unload.

    max_tokens is the desired *output* budget. THINK_BUDGET is added
    automatically inside _generate_with for thinking models.
    """
    system = build_supervisor_answer_system(
        (
        "You are a helpful, knowledgeable assistant. "
        "Answer clearly and concisely."
        ),
        "general topics",
    )
    return _generate_supervisor_answer_resilient(
        mgr=mgr,
        prompt=query,
        system=system,
        max_tokens=OUTPUT_BUDGET_T1,
        temperature=0.7,
        think_budget=THINK_BUDGET_T1,
        retry_label="tier1",
    )


# ── TIER2 Pipeline ────────────────────────────────────────────────────────

def run_tier2(
    query: str,
    domain: str,
    mgr: ModelManager,
    session_store: "SessionStore",
    template_store: "TemplateStore",
    enhance: bool = True,
    kb_enabled: bool = True,
    render_mode: str = "intermediate+final",
    resume_after_switch: bool = False,
) -> str:
    """
        TIER2 benchmark-style pipeline.
      0  Extract qualifiers + patch persona
      1  KB pre-check (supervisor decides)
            2  Ensure specialist + supervisor prompt formulation
            3  Specialist draft generation
            4  Quality-gated single retry (if needed)
            5  Keep better specialist candidate
            6  Supervisor synthesis and keep better final answer
    """
    show_intermediate = _normalize_render_mode(render_mode) != "final only"
    task_key = session_store.peek_task_key() if session_store else f"adhoc:{int(time.time())}"
    phase_artifact_ids: Dict[str, str] = {}

    # Step 0 — Qualifier extraction
    qualifiers = extract_qualifiers(query)
    specialist_system = patch_persona(domain, qualifiers)
    supervisor_answer_system = build_supervisor_answer_system(specialist_system, domain)
    supervisor_tier2_system = build_supervisor_tier2_specialist_system(specialist_system, domain)
    if qualifiers.has_any():
        cprint(f"[CoE] Qualifiers: {qualifiers.summary()}", "dim")

    def _fallback_specialist_task_prompt(
        kb_context: Optional[str] = None,
        template_match=None,
    ) -> str:
        template_injection = ""
        if template_match:
            imperative = TEMPLATE_IMPERATIVE[template_match.strength]
            template_injection = f"\n\n{imperative}\n{template_match.scaffold_text}"
        ctx_block = f"\n\nSession context:\n{session_ctx}" if session_ctx else ""
        kb_block = f"\n\nKB reference:\n{kb_context}" if kb_context else ""
        return f"{query}{ctx_block}{kb_block}{template_injection}".strip()

    def _looks_like_prompt_scaffold(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return True
        suspicious_markers = [
            "output only the prompt text",
            "construct an optimal prompt",
            "context for code specialist",
            "insert missing context here",
            "task:",
            "instructions:",
        ]
        if lowered.startswith("prompt"):
            return True
        return sum(1 for marker in suspicious_markers if marker in lowered) >= 2

    def _run_zero_shot_supervisor_specialist(
        reason: str,
        kb_context: Optional[str] = None,
        template_match=None,
    ) -> str:
        cprint(f"[LoadPolicy] {reason} — defaulting to supervisor-only zero-shot specialist harness.", "yellow")
        template_injection = ""
        if template_match:
            imperative = TEMPLATE_IMPERATIVE[template_match.strength]
            template_injection = f"\n\n{imperative}\n{template_match.scaffold_text}"
        ctx_block = f"\n\nSession context:\n{session_ctx}" if session_ctx else ""
        kb_block = f"\n\nReference:\n{kb_context}" if kb_context else ""
        task_prompt = (
            f"Answer the user's request directly as the {domain} specialist."
            f" Do not write meta-instructions, prompt text, or analysis headers."
            f"\n\nUser request: {query}"
            f"{ctx_block}{kb_block}{template_injection}"
        )
        # Synthetic supervisor-only path: take a single substantive answer
        # pass with the normal TIER2 think budget, but do not continue into
        # the later TIER2 grading/retry/synthesis machinery.
        result = _generate_supervisor_answer_resilient(
            mgr=mgr,
            prompt=task_prompt,
            system=supervisor_tier2_system,
            max_tokens=OUTPUT_BUDGET_T2,
            temperature=0.2,
            think_budget=THINK_BUDGET_T2,
            retry_label=f"tier2_synthetic_{domain}",
        )
        if session_store:
            artifacts = {"generation_artifact_ids": phase_artifact_ids}
            tier = "TIER2_SYNTHETIC"
            session_store.write_task(query, tier, [domain], result, artifacts)
        return result

    # Session context
    session_ctx = session_store.get_context_for_query(query) if session_store else ""

    # Template pre-check
    match = None
    if template_store and mgr._embedder:
        match = template_store.find_match(query, mgr._embedder)
        if match:
            cprint(f"[TEMPLATE] {match.strength.upper()} match: {match.title} ({match.similarity:.2f})")

    if domain == "supervisor":
        cprint("[LoadPolicy] Supervisor-routed TIER2 request — answering directly with supervisor.", "dim")
        template_injection = ""
        if match:
            imperative = TEMPLATE_IMPERATIVE[match.strength]
            template_injection = f"\n\n{imperative}\n{match.scaffold_text}"
        ctx_block = f"\n\nSession context:\n{session_ctx}" if session_ctx else ""
        task_prompt = (
            "Answer the user's request directly. "
            "Do not write meta-instructions, prompt text, or analysis headers."
            f"\n\nUser request: {query}"
            f"{ctx_block}{template_injection}"
        )
        # Direct supervisor route: single substantive answer pass with the
        # normal TIER2 think budget, but no later TIER2 grading/retry logic.
        result = _generate_supervisor_answer_resilient(
            mgr=mgr,
            prompt=task_prompt,
            system=supervisor_tier2_system,
            max_tokens=OUTPUT_BUDGET_T2,
            temperature=0.2,
            think_budget=THINK_BUDGET_T2,
            retry_label=f"tier2_supervisor_{domain}",
        )
        if session_store:
            artifacts = {"generation_artifact_ids": phase_artifact_ids}
            session_store.write_task(query, "TIER2", [domain], result, artifacts)
        return result

    # Step 1 — Benchmark-style direct specialist prompt path.
    # Avoid pre-specialist supervisor calls; they have proven unstable across
    # domain transitions in this long-lived CLI process.
    kb_context = None
    if resume_after_switch:
        cprint("[LoadPolicy] Benchmark-style resumed switch: skipping supervisor pre-processing before specialist load.", "dim")

    synthetic_reason = ""

    # Step 2 — Background specialist load + supervisor prompt formulation
    # 'supervisor' domain always runs through the supervisor directly (no separate specialist)
    if domain in DISABLED_SPECIALIST_DOMAINS:
        cprint(f"[LoadPolicy] {domain} specialist disabled on this build — using supervisor-driven synthetic specialist.", "yellow")
    _raw = DOMAIN_MODELS.get(domain) if domain != "supervisor" and domain not in DISABLED_SPECIALIST_DOMAINS else None
    specialist_model = mgr._resolve(_raw) if _raw else None
    is_synthetic = specialist_model is None
    defer_specialist_load = False

    if is_synthetic:
        synthetic_reason = f"no domain model available for '{domain}'"
        cprint(f"[SYNTHETIC SPECIALIST — {synthetic_reason}]", "yellow")
    else:
        current_specialist = mgr.specialist_path()
        if current_specialist and current_specialist != specialist_model:
            defer_specialist_load = True
            cprint(
                f"[LoadPolicy] Serializing specialist switch for {Path(specialist_model).name} "
                f"until supervisor formulation completes.",
                "dim",
            )
            debug_log(
                "specialist.load_deferred",
                current=Path(current_specialist).name,
                target=Path(specialist_model).name,
                reason="cross-domain-switch",
            )
        else:
            mgr.ensure_specialist_for_model(specialist_model)

    if enhance and not resume_after_switch and is_synthetic:
        if is_synthetic:
            return _run_zero_shot_supervisor_specialist(synthetic_reason or f"synthetic {domain} path", kb_context, match)
        template_injection = ""
        if match:
            imperative = TEMPLATE_IMPERATIVE[match.strength]
            template_injection = f"\n\n{imperative}\n{match.scaffold_text}"

        ctx_block = f"\n\nSession context:\n{session_ctx}" if session_ctx else ""
        kb_block = f"\n\nKB reference:\n{kb_context}" if kb_context else ""
        qual_note = f" Specialist qualifiers detected: {qualifiers.summary()}." if qualifiers.has_any() else ""

        formulation_system = (
            f"You are a precise prompt engineer. Construct an optimal prompt for a {domain} "
            f"specialist to answer the following query.{qual_note} Incorporate any provided context. "
            f"Output only the prompt text."
        )
        formulation_input = (
            f"Original query: {query}"
            f"{ctx_block}{kb_block}{template_injection}"
        )
        enriched_prompt = mgr.generate_supervisor(
            formulation_input, system=formulation_system, max_tokens=512, temperature=0.2,
            think_budget=THINK_BUDGET_UTIL,
        )
        if _looks_like_prompt_scaffold(enriched_prompt):
            cprint("[LoadPolicy] Discarding prompt-scaffold enhancement and using raw specialist task prompt.", "yellow")
            debug_log("tier2.prompt_scaffold_discarded", domain=domain)
            enriched_prompt = _fallback_specialist_task_prompt(kb_context, match)
    else:
        if is_synthetic:
            return _run_zero_shot_supervisor_specialist(synthetic_reason or f"synthetic {domain} path", kb_context, match)
        enriched_prompt = _fallback_specialist_task_prompt(kb_context, match)

    if resume_after_switch and not is_synthetic:
        enriched_prompt = _fallback_specialist_task_prompt(kb_context, match)

    if not is_synthetic and defer_specialist_load:
        mgr.ensure_specialist_for_model(specialist_model)

    if domain == "code":
        code_language = qualifiers.language or "requested"
        if _wants_code_only(query):
            enriched_prompt += (
                f"\n\nOutput requirements: Return ONLY one {code_language} function definition. "
                "No prose, no explanation, no markdown fences, no comments, no docstring."
            )
        else:
            enriched_prompt += (
                f"\n\nOutput requirements: Start with a valid {code_language} function definition. "
                "Keep any extra explanation minimal."
            )

    if not is_synthetic:
        mgr.await_specialist()
        if mgr._load_error:
            cprint(f"[TIER2] Specialist load failed: {mgr._load_error} — using synthetic path", "yellow")
            is_synthetic = True
            return _run_zero_shot_supervisor_specialist(f"specialist load failed for '{domain}'", kb_context, match)

    # Step 3 — Specialist draft generation
    if is_synthetic:
        draft = mgr.generate_supervisor(enriched_prompt, system=supervisor_tier2_system,
                                        max_tokens=OUTPUT_BUDGET_T2, temperature=0.3,
                                        think_budget=THINK_BUDGET_T2)
    else:
        draft = mgr.generate_specialist(enriched_prompt, system=specialist_system,
                                        max_tokens=OUTPUT_BUDGET_T2)
    if domain == "code":
        draft = _normalize_python_function_output(query, draft)

    # Show raw generation immediately so users can always inspect model output
    # even if later grading/synthesis stages fail at native runtime.
    if show_intermediate:
        render_response(draft, title="Response (Draft)")

    # Step 4 — Quality evaluation
    grade_system = "You are a strict technical evaluator. Output only a verdict line."
    if (not is_synthetic) and domain in SPECIALIST_SELF_GRADE_DOMAINS:
        draft_verdict_word, draft_verdict_line = _grade_with_specialist_self_check(query, draft, domain, mgr)
    else:
        draft_verdict_word, draft_verdict_line = _grade_output(query, draft, domain, mgr, grade_system)
    cprint(f"[Grade] {draft_verdict_line[:80]}", "dim")

    draft_artifact_id = ""
    if session_store:
        draft_artifact_id = session_store.add_generation_artifact(
            task_key, "tier2_draft", draft, domain=domain, verdict=draft_verdict_word,
        )
        phase_artifact_ids["draft"] = draft_artifact_id

    best_specialist = draft
    best_specialist_word = draft_verdict_word
    best_specialist_line = draft_verdict_line
    best_specialist_artifact_id = draft_artifact_id

    # Step 5 — Single retry when specialist output fails quality gate
    if _should_retry_tier2(draft_verdict_word, draft):
        fail_cat = draft_verdict_word if draft_verdict_word in FAIL_CATEGORIES else "FAIL_INCOMPLETE"
        fail_reason = draft_verdict_line[len(fail_cat):].lstrip(":").strip() if ":" in draft_verdict_line else "short output"
        cprint(f"[TIER2] Retry — {fail_cat}: {fail_reason[:60]}", "yellow")

        retry_kb = None
        if kb_enabled:
            retry_kb = kb_retrieve_retry(f"{fail_reason} {query}", domain)

        retry_guidance = _build_retry_guidance(
            query=query,
            task_prompt=enriched_prompt,
            fail_category=fail_cat,
            fail_reason=fail_reason,
            mgr=mgr,
            kb_context=retry_kb,
        )

        retry_parts = [
            "Retry this task using the guidance below.",
            f"\nGuidance:\n{retry_guidance}",
        ]
        if retry_kb:
            retry_parts.append(f"\nAdditional reference:\n{retry_kb}")
        retry_parts.append(f"\nTask: {enriched_prompt}")
        retry_prompt = "\n".join(retry_parts)

        if is_synthetic:
            retry_draft = mgr.generate_supervisor(retry_prompt, system=supervisor_tier2_system,
                                                  max_tokens=OUTPUT_BUDGET_T2, temperature=0.4,
                                                  think_budget=THINK_BUDGET_T2)
        else:
            if not mgr.specialist_path():
                mgr.ensure_specialist_for_model(specialist_model)
                mgr.await_specialist()
            retry_draft = mgr.generate_specialist(retry_prompt, system=specialist_system,
                                                  max_tokens=OUTPUT_BUDGET_T2, temperature=0.4)
        if domain == "code":
            retry_draft = _normalize_python_function_output(query, retry_draft)

        if show_intermediate:
            render_response(retry_draft, title="Response (Retry)")

        if (not is_synthetic) and domain in SPECIALIST_SELF_GRADE_DOMAINS:
            retry_verdict_word, retry_verdict_line = _grade_with_specialist_self_check(query, retry_draft, domain, mgr)
        else:
            retry_verdict_word, retry_verdict_line = _grade_output(query, retry_draft, domain, mgr, grade_system)
        cprint(f"[Grade retry] {retry_verdict_line[:80]}", "dim")

        retry_artifact_id = ""
        if session_store:
            retry_artifact_id = session_store.add_generation_artifact(
                task_key, "tier2_retry", retry_draft, domain=domain, verdict=retry_verdict_word,
            )
            phase_artifact_ids["retry"] = retry_artifact_id

        best_specialist, best_specialist_word, best_specialist_line = _pick_better_candidate(
            draft,
            draft_verdict_word,
            draft_verdict_line,
            retry_draft,
            retry_verdict_word,
            retry_verdict_line,
        )
        chosen = "retry" if best_specialist is retry_draft else "draft"
        best_specialist_artifact_id = retry_artifact_id if chosen == "retry" else draft_artifact_id
        cprint(f"[TIER2] Kept better specialist candidate: {chosen} ({best_specialist_word})", "dim")

    if not is_synthetic and mgr.specialist_path():
        debug_log(
            "specialist.unload_for_supervisor",
            model=Path(mgr.specialist_path()).name,
            reason="post-specialist-selection-benchmark-mode",
        )
        mgr.unload_specialist()

    # Step 6 — Supervisor synthesis and keep-better selection
    result = best_specialist
    result_word = best_specialist_word
    result_line = best_specialist_line
    result_artifact_id = best_specialist_artifact_id
    if ENABLE_TIER2_SYNTHESIS and best_specialist_word != "PASS":
        synthesis_system = (
            "You are a brilliant coordinator. A domain specialist produced the following draft. "
            "Refine it into a precise, complete, well-structured final answer."
        )
        specialist_for_synthesis = (
            session_store.get_generation_text(best_specialist_artifact_id)
            if (session_store and best_specialist_artifact_id)
            else best_specialist
        )
        # Skeleton = full prompt with draft absent; measures tokens available for draft.
        _synth_skeleton = (
            f"Original question: {query}\n\nSpecialist draft:\n\nProvide the refined final answer:"
        )
        draft_fitted = _fit_draft(
            specialist_for_synthesis, _synth_skeleton, mgr,
            OUTPUT_BUDGET_SYNTHESIS, THINK_BUDGET_T2,
            system=synthesis_system,
            max_draft_tokens=SYNTHESIS_DRAFT_TOKEN_CAP,
        )
        synthesis_prompt = (
            f"Original question: {query}\n\nSpecialist draft:\n{draft_fitted}\n\nProvide the refined final answer:"
        )
        synth_result = mgr.generate_supervisor(
            synthesis_prompt, system=synthesis_system,
            max_tokens=OUTPUT_BUDGET_SYNTHESIS, temperature=0.2,
            think_budget=THINK_BUDGET_T2,
        )
        if domain == "code":
            synth_result = _normalize_python_function_output(query, synth_result)

        if show_intermediate:
            render_response(synth_result, title="Response (Synthesis)")

        synth_verdict_word, synth_verdict_line = _grade_output(query, synth_result, domain, mgr, grade_system)
        cprint(f"[Grade synthesis] {synth_verdict_line[:80]}", "dim")

        synth_artifact_id = ""
        if session_store:
            synth_artifact_id = session_store.add_generation_artifact(
                task_key, "tier2_synthesis", synth_result, domain=domain, verdict=synth_verdict_word,
            )
            phase_artifact_ids["synthesis"] = synth_artifact_id

        result, result_word, result_line = _pick_better_candidate(
            best_specialist,
            best_specialist_word,
            best_specialist_line,
            synth_result,
            synth_verdict_word,
            synth_verdict_line,
        )
        source = "synthesis" if result is synth_result else "specialist"
        result_artifact_id = synth_artifact_id if source == "synthesis" else best_specialist_artifact_id
        cprint(f"[TIER2] Final source: {source} ({result_word})", "dim")
    elif ENABLE_TIER2_SYNTHESIS:
        cprint("[TIER2] Synthesis bypassed — specialist candidate already PASS.", "dim")
    else:
        cprint("[TIER2] Synthesis bypassed — returning best specialist candidate.", "dim")

    final_word, final_line = result_word, result_line
    cprint(f"[Grade final] {final_line[:80]}", "dim")

    if _should_trigger_supervisor_fallback(final_word, domain):
        cprint("[TIER2] Score-gated fallback: supervisor direct pass", "yellow")
        fallback = _run_supervisor_fallback(
            query=query,
            mgr=mgr,
            current_answer=result,
            max_tokens=OUTPUT_BUDGET_T2,
            think_budget=THINK_BUDGET_T2,
        )
        if domain == "code":
            fallback = _normalize_python_function_output(query, fallback)
        if _normalize_render_mode(render_mode) != "final only":
            render_response(fallback, title="Response (Fallback)")

        fb_word, fb_line = _grade_output(query, fallback, domain, mgr, grade_system)
        cprint(f"[Grade fallback] {fb_line[:80]}", "dim")
        fallback_artifact_id = ""
        if session_store:
            fallback_artifact_id = session_store.add_generation_artifact(
                task_key, "tier2_fallback", fallback, domain=domain, verdict=fb_word,
            )
            phase_artifact_ids["fallback"] = fallback_artifact_id
        result, result_word, _ = _pick_better_candidate(
            result,
            final_word,
            final_line,
            fallback,
            fb_word,
            fb_line,
        )
        picked = "fallback" if result is fallback else "pipeline"
        if picked == "fallback":
            result_artifact_id = fallback_artifact_id
        cprint(f"[TIER2] Post-fallback source: {picked} ({result_word})", "dim")

    # Persist
    if session_store:
        artifacts = {}
        if match:
            artifacts["template"] = match.template_id
        artifacts["generation_artifact_ids"] = phase_artifact_ids
        artifacts["final_artifact_id"] = result_artifact_id
        tier = "TIER2_SYNTHETIC" if is_synthetic else "TIER2"
        session_store.write_task(query, tier, [domain], result, artifacts)

    return result


# ═══════════════════════════════════════════════════════════════════════════
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
    """Wraps MemoryBackbone for cross-task session persistence."""

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

        if MEMORY_AVAILABLE:
            db_path = _HERE / "data" / "memory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._db = MemoryBackbone(MemoryConfig(db_path=db_path))
            except Exception as e:
                cprint(f"[SessionStore] Could not init DB: {e}", "yellow")

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

    def write_task(
        self,
        query: str,
        tier: str,
        domains: List[str],
        output: str,
        artifacts: dict,
    ) -> None:
        key = f"{self.session_id}:task:{self.task_seq:04d}"
        value = {
            "task_key": key,
            "query": query,
            "tier": tier,
            "domains": domains,
            "output_snippet": output[:300],
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

    def get_context_for_query(self, query: str) -> str:
        if self.task_seq == 0:
            return ""
        try:
            recent = self.get_recent_tasks(2)
            if not recent:
                return ""
            lines = ["=== Session Context (last 2 tasks) ==="]
            for v in recent:
                t = v.get("tier", "?")
                dom = ",".join(v.get("domains", []))
                qsnip = v.get("query", "")[:120]
                osnip = v.get("output_snippet", "")
                lines.append(f"[{t} | {dom}] Query: {qsnip}")
                if osnip:
                    lines.append(f"Output snippet: {osnip[:220]}")

                arts = v.get("artifacts", {}) or {}
                final_id = arts.get("final_artifact_id") if isinstance(arts, dict) else None
                if final_id:
                    final_text = self.get_generation_text(final_id)
                    if final_text:
                        lines.append(f"Accepted final excerpt: {final_text[:320]}")
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

    def get_stats(self) -> dict:
        return {
            "session_id": self.session_id,
            "task_count": self.task_seq,
            "artifact_count": self._artifacts.count(),
            "pending_switch": self._pending_switch.get("to_domain") if self._pending_switch else None,
            "db_available": self._db is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
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
# PHASE 6 — TIER3 Pipeline with Step Buffer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
    step_idx: int
    domain: str
    summary: str
    artifacts: dict
    full_output: str
    output_artifact_id: str = ""


class StepBuffer:
    def __init__(self) -> None:
        self._records: List[StepRecord] = []

    def add(self, record: StepRecord) -> None:
        self._records.append(record)

    def get_all(self) -> List[StepRecord]:
        return list(self._records)

    def query_for_step(self, current_domain: str) -> str:
        if not self._records:
            return ""
        lines = ["=== Prior Step Results ==="]
        for r in self._records:
            arts = r.artifacts
            ents = arts.get("entities", [])
            decs = arts.get("decisions", [])
            cons = arts.get("constraints", [])
            lines.append(
                f"Step {r.step_idx} ({r.domain}): "
                f"entities={ents}, decisions={decs}, constraints={cons}"
            )
        lines.append("===")
        return "\n".join(lines)

    def merged_artifacts(self) -> dict:
        all_ents, all_decs, all_cons = [], [], []
        for r in self._records:
            all_ents.extend(r.artifacts.get("entities", []))
            all_decs.extend(r.artifacts.get("decisions", []))
            all_cons.extend(r.artifacts.get("constraints", []))
        return {"entities": all_ents, "decisions": all_decs, "constraints": all_cons}


def _supervisor_extract(output: str, mgr: ModelManager) -> dict:
    """Extract entities/decisions/constraints from step output via supervisor."""
    prompt = (
        "Extract from the following output as JSON with keys: "
        "entities (list of named things), decisions (list of choices made), "
        "constraints (list of requirements or limits).\n\n"
        f"{output}"
    )
    system = "You are a precise extraction engine. Output only valid JSON."
    try:
        raw = mgr.generate_supervisor(prompt, system=system, max_tokens=200, temperature=0.01, think_budget=THINK_BUDGET_UTIL)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {"entities": [], "decisions": [], "constraints": []}


def run_tier3(
    query: str,
    domains: List[str],
    mgr: ModelManager,
    session_store: "SessionStore",
    template_store: "TemplateStore",
    kb_enabled: bool = True,
) -> str:
    """TIER3: sequential multi-specialist pipeline with step buffer."""
    buffer = StepBuffer()
    task_key = session_store.peek_task_key() if session_store else f"adhoc:{int(time.time())}"
    phase_artifact_ids: Dict[str, str] = {}
    session_ctx = session_store.get_context_for_query(query) if session_store else ""
    grade_system = "You are a strict technical evaluator. Output only a verdict line."

    # Check 1 — Whole-query template match
    whole_match = None
    slot_hint = ""
    if template_store and mgr._embedder:
        whole_match = template_store.find_match(query, mgr._embedder)
        if whole_match:
            cprint(
                f"[TEMPLATE-TIER3] Whole-query {whole_match.strength} match: "
                f"{whole_match.title} ({whole_match.similarity:.2f}) — "
                f"{len(whole_match.slot_sequence)} slots advisory"
            )
            slot_hint = (
                f"\nA related multi-domain template suggests this step sequence: "
                f"{whole_match.slot_sequence}. Use as advisory ordering — "
                f"do not override the classified domains."
            )

    # Step loop
    for i, domain in enumerate(domains):
        cprint(f"\n[TIER3 step {i+1}/{len(domains)}] domain={domain}")

        # Prepare specialist transition early so cross-domain steps return to a
        # supervisor-only neutral state before the next specialist is loaded.
        if domain in DISABLED_SPECIALIST_DOMAINS:
            cprint(f"[LoadPolicy] {domain} specialist disabled on this build — using supervisor-driven synthetic specialist.", "yellow")
        _raw = DOMAIN_MODELS.get(domain) if domain != "supervisor" and domain not in DISABLED_SPECIALIST_DOMAINS else None
        specialist_model = mgr._resolve(_raw) if _raw else None
        is_synthetic = specialist_model is None
        current_specialist = mgr.specialist_path()
        needs_reset_barrier = bool(
            current_specialist and specialist_model and current_specialist != specialist_model
        )
        if needs_reset_barrier:
            cprint("[LoadPolicy] TIER3 hidden reset barrier: returning to supervisor-only state before next specialist.", "dim")
            debug_log(
                "tier3.reset_barrier.begin",
                step=i + 1,
                current=Path(current_specialist).name,
                target=Path(specialist_model).name,
            )
            mgr.reset_to_supervisor_only(reason=f"tier3-step-{i+1}-barrier")
            try:
                current_specialist = mgr.specialist_path()
            except Exception as e:
                cprint(f"[LoadPolicy] TIER3 reset barrier warning: {e}", "dim")
                debug_log(
                    "tier3.reset_barrier.warning",
                    step=i + 1,
                    error=str(e),
                )
            else:
                debug_log(
                    "tier3.reset_barrier.end",
                    step=i + 1,
                    target=Path(specialist_model).name,
                )

        # Check 2 — Per-step template match
        step_desc = f"{domain}: addressing part of the query: {query[:100]}"
        step_match = None
        if template_store and mgr._embedder:
            step_match = template_store.find_match_for_step(step_desc, domain, mgr._embedder)
            if step_match:
                cprint(f"[TEMPLATE-STEP {i+1}] {step_match.strength} match: {step_match.title} ({step_match.similarity:.2f})")

        buffer_ctx = buffer.query_for_step(domain)

        # Qualifiers for this step
        qualifiers = extract_qualifiers(query)
        specialist_system = patch_persona(domain, qualifiers)

        # Injections
        template_injection = ""
        if step_match:
            imperative = TEMPLATE_IMPERATIVE[step_match.strength]
            template_injection = f"\n\n{imperative}\n{step_match.scaffold_text}"

        # Supervisor step formulation
        formulation_system = (
            f"You are a precise prompt engineer constructing a step-specific prompt "
            f"for a {domain} specialist as part of a multi-domain task."
        )
        formulation_input = (
            f"Full task: {query}"
            f"{slot_hint}"
            f"\n\nSession context:\n{session_ctx}" if session_ctx else f"Full task: {query}{slot_hint}"
        )
        if buffer_ctx:
            formulation_input += f"\n\n{buffer_ctx}"
        formulation_input += f"\n\nYour job: address the {domain} component only.{template_injection}"

        step_prompt = mgr.generate_supervisor(
            formulation_input, system=formulation_system, max_tokens=512, temperature=0.2,
            think_budget=THINK_BUDGET_UTIL,
        )

        # Background specialist load
        # 'supervisor' domain always runs through the supervisor directly
        switched_specialist_safemode = False
        if not is_synthetic:
            current_specialist = mgr.specialist_path()
            if current_specialist and current_specialist != specialist_model:
                switched_specialist_safemode = True
            mgr.ensure_specialist_for_model(specialist_model)
            mgr.await_specialist()
            if mgr._load_error:
                is_synthetic = True

        # Generate
        if is_synthetic:
            step_output = mgr.generate_supervisor(step_prompt, system=specialist_system,
                                                  max_tokens=OUTPUT_BUDGET_T3, temperature=0.3,
                                                  think_budget=THINK_BUDGET_T3)
        else:
            step_output = mgr.generate_specialist(step_prompt, system=specialist_system,
                                                  max_tokens=OUTPUT_BUDGET_T3)
            if switched_specialist_safemode and mgr.specialist_path():
                cprint("[LoadPolicy] Unloading switched specialist before TIER3 supervisor post-processing.", "dim")
                debug_log(
                    "specialist.unload_for_supervisor",
                    model=Path(mgr.specialist_path()).name,
                    reason=f"tier3-step-{i+1}-post-draft-safemode",
                )
                mgr.unload_specialist()
        if domain == "code":
            step_output = _normalize_python_function_output(query, step_output)

        # Grade + benchmark-style single retry gate
        verdict_word, verdict_line = _grade_output(query, step_output, domain, mgr, grade_system)
        cprint(f"[Grade step {i+1}] {verdict_line[:80]}", "dim")

        draft_artifact_id = ""
        if session_store:
            draft_artifact_id = session_store.add_generation_artifact(
                task_key, f"tier3_step{i+1}_draft", step_output, domain=domain, verdict=verdict_word,
            )
            phase_artifact_ids[f"step{i+1}_draft"] = draft_artifact_id

        best_step_output = step_output
        best_step_word = verdict_word
        best_step_line = verdict_line
        best_step_artifact_id = draft_artifact_id

        if _should_retry_tier2(verdict_word, step_output):
            fail_reason = verdict_line[len(verdict_word):].lstrip(":").strip()
            retry_kb = None
            if kb_enabled:
                retry_kb = kb_retrieve_retry(f"{fail_reason} {query}", domain)

            retry_guidance = _build_retry_guidance(
                query=query,
                task_prompt=step_prompt,
                fail_category=verdict_word,
                fail_reason=fail_reason,
                mgr=mgr,
                kb_context=retry_kb,
            )

            retry_parts = [
                f"Issue: {fail_reason}",
                f"Guidance:\n{retry_guidance}",
            ]
            if retry_kb:
                retry_parts.append(f"Additional reference:\n{retry_kb}")
            retry_parts.append(f"Revised task:\n{step_prompt}")
            retry_prompt = "\n".join(retry_parts)
            if is_synthetic:
                retry_output = mgr.generate_supervisor(retry_prompt, system=specialist_system,
                                                       max_tokens=OUTPUT_BUDGET_T3, temperature=0.4,
                                                       think_budget=THINK_BUDGET_T3)
            else:
                if switched_specialist_safemode and not mgr.specialist_path():
                    mgr.ensure_specialist_for_model(specialist_model)
                    mgr.await_specialist()
                retry_output = mgr.generate_specialist(retry_prompt, system=specialist_system,
                                                       max_tokens=OUTPUT_BUDGET_T3, temperature=0.4)
                if switched_specialist_safemode and mgr.specialist_path():
                    cprint("[LoadPolicy] Unloading switched specialist before TIER3 retry grading.", "dim")
                    debug_log(
                        "specialist.unload_for_supervisor",
                        model=Path(mgr.specialist_path()).name,
                        reason=f"tier3-step-{i+1}-post-retry-safemode",
                    )
                    mgr.unload_specialist()
            if domain == "code":
                retry_output = _normalize_python_function_output(query, retry_output)

            retry_word, retry_line = _grade_output(query, retry_output, domain, mgr, grade_system)
            cprint(f"[Grade step {i+1} retry] {retry_line[:80]}", "dim")

            retry_artifact_id = ""
            if session_store:
                retry_artifact_id = session_store.add_generation_artifact(
                    task_key, f"tier3_step{i+1}_retry", retry_output, domain=domain, verdict=retry_word,
                )
                phase_artifact_ids[f"step{i+1}_retry"] = retry_artifact_id

            best_step_output, best_step_word, best_step_line = _pick_better_candidate(
                step_output,
                verdict_word,
                verdict_line,
                retry_output,
                retry_word,
                retry_line,
            )
            picked = "retry" if best_step_output is retry_output else "draft"
            best_step_artifact_id = retry_artifact_id if picked == "retry" else draft_artifact_id
            cprint(f"[TIER3 step {i+1}] kept {picked} ({best_step_word})", "dim")

        step_output = (
            session_store.get_generation_text(best_step_artifact_id)
            if (session_store and best_step_artifact_id)
            else best_step_output
        )
        if best_step_artifact_id:
            phase_artifact_ids[f"step{i+1}_accepted"] = best_step_artifact_id

        # Domain-sticky specialist policy: do not unload here.
        # Specialist is unloaded only when switching domains/models, on error,
        # or during shutdown.

        # Extract + buffer
        extracted = _supervisor_extract(step_output, mgr)
        buffer.add(StepRecord(
            step_idx=i + 1,
            domain=domain,
            summary=extracted.get("entities", [""])[0] if extracted.get("entities") else "",
            artifacts=extracted,
            full_output=step_output,
            output_artifact_id=best_step_artifact_id,
        ))
        cprint(f"[TIER3 step {i+1}/{len(domains)}] {domain} complete ✓")

    # Final synthesis
    if buffer.get_all() and session_store and buffer.get_all()[-1].output_artifact_id:
        last_output = session_store.get_generation_text(buffer.get_all()[-1].output_artifact_id)
    else:
        last_output = buffer.get_all()[-1].full_output if buffer.get_all() else ""
    step_summaries = buffer.query_for_step("")

    synthesis_system = (
        "You are a senior architect synthesizing outputs from multiple domain specialists "
        "into a single, coherent, complete response."
    )
    # Skeleton = full prompt with last_output absent; measures tokens available for it.
    _synth_skeleton = (
        f"Original question: {query}\n\n"
        f"{session_ctx}\n\n{step_summaries}\n\n"
        f"Last specialist output:\n\n"
        f"Provide the final synthesized answer:"
    )
    last_fitted = _fit_draft(
        last_output, _synth_skeleton, mgr,
        OUTPUT_BUDGET_SYNTHESIS, THINK_BUDGET_T2,
        system=synthesis_system,
        max_draft_tokens=SYNTHESIS_DRAFT_TOKEN_CAP,
    )
    synthesis_prompt = (
        f"Original question: {query}\n\n"
        f"{session_ctx}\n\n{step_summaries}\n\n"
        f"Last specialist output:\n{last_fitted}\n\n"
        f"Provide the final synthesized answer:"
    )
    final = mgr.generate_supervisor(synthesis_prompt, system=synthesis_system,
                                    max_tokens=OUTPUT_BUDGET_SYNTHESIS, temperature=0.2,
                                    think_budget=THINK_BUDGET_T2)
    final_artifact_id = ""

    final_word, final_line = _grade_output(query, final, "supervisor", mgr, grade_system)
    cprint(f"[Grade TIER3 final] {final_line[:80]}", "dim")
    if session_store:
        final_artifact_id = session_store.add_generation_artifact(
            task_key, "tier3_synthesis", final, domain="supervisor", verdict=final_word,
        )
        phase_artifact_ids["synthesis"] = final_artifact_id

    if _should_trigger_supervisor_fallback(final_word, "multi"):
        cprint("[TIER3] Score-gated fallback: supervisor direct pass", "yellow")
        fallback = _run_supervisor_fallback(
            query=query,
            mgr=mgr,
            current_answer=final,
            max_tokens=OUTPUT_BUDGET_SYNTHESIS,
            think_budget=THINK_BUDGET_T3,
        )
        fb_word, fb_line = _grade_output(query, fallback, "supervisor", mgr, grade_system)
        cprint(f"[Grade TIER3 fallback] {fb_line[:80]}", "dim")
        fallback_artifact_id = ""
        if session_store:
            fallback_artifact_id = session_store.add_generation_artifact(
                task_key, "tier3_fallback", fallback, domain="supervisor", verdict=fb_word,
            )
            phase_artifact_ids["fallback"] = fallback_artifact_id
        final, kept_word, _ = _pick_better_candidate(
            final,
            final_word,
            final_line,
            fallback,
            fb_word,
            fb_line,
        )
        if final is fallback:
            final_artifact_id = fallback_artifact_id
        source = "fallback" if final is fallback else "synthesis"
        cprint(f"[TIER3] Final source: {source} ({kept_word})", "dim")

    if session_store:
        task_artifacts = buffer.merged_artifacts()
        task_artifacts["generation_artifact_ids"] = phase_artifact_ids
        task_artifacts["final_artifact_id"] = final_artifact_id
        session_store.write_task(query, "TIER3", domains, final, task_artifacts)

    return final


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7 — Knowledge Base Integration
# ═══════════════════════════════════════════════════════════════════════════

_kb_instance: Optional[object] = None
_kb_embedder_fn = None


def _kb_search(query: str, domain: str) -> Optional[str]:
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
    return _kb_search(query, domain)


def kb_retrieve_retry(targeted_query: str, domain: str) -> Optional[str]:
    return _kb_search(targeted_query, domain)


def init_kb(embedder: "EmbeddingManager") -> None:
    global _kb_embedder_fn
    if embedder and hasattr(embedder, "encode"):
        _kb_embedder_fn = embedder.encode


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8 — CLI Shell
# ═══════════════════════════════════════════════════════════════════════════

def _settings_table(settings: dict, sources: dict) -> None:
    if RICH_AVAILABLE and console:
        tbl = Table(title="[bold]College of Experts — Settings[/bold]", show_lines=True)
        tbl.add_column("Setting", style="cyan")
        tbl.add_column("Value")
        tbl.add_column("Source", style="dim")
        tbl.add_column("Mutable", style="dim")
        mutability = {
            "device":  "restart-only",
            "session": "/new to rotate",
            "enhance": "/enhance on|off",
            "confirm": "/confirm on|off",
            "kb":      "/kb on|off",
            "render_mode": "/render intermediate|final",
            "embed_cpu_batch_size": "restart-only",
            "embed_cpu_threads": "restart-only",
            "max_runtime_seq_tokens": "restart-only",
        }
        for k in ["device", "session", "enhance", "confirm", "kb", "render_mode", "embed_cpu_batch_size", "embed_cpu_threads", "max_runtime_seq_tokens"]:
            v = settings.get(k)
            display = str(v) if v is not None else "(auto)"
            console.print(
                tbl.add_column if False else None  # just build inline
            )
            tbl.add_row(k, display, sources.get(k, "default"), mutability.get(k, ""))
        console.print(tbl)
    else:
        print(f"\n{'Setting':<12} {'Value':<20} {'Source':<16} {'Mutable'}")
        print("─" * 64)
        for k in ["device", "session", "enhance", "confirm", "kb", "render_mode", "embed_cpu_batch_size", "embed_cpu_threads", "max_runtime_seq_tokens"]:
            v = settings.get(k)
            display = str(v) if v is not None else "(auto)"
            src = sources.get(k, "default")
            mut = {"device": "restart-only", "session": "/new to rotate",
                   "enhance": "/enhance on|off", "confirm": "/confirm on|off",
                   "kb": "/kb on|off", "render_mode": "/render intermediate|final",
                   "embed_cpu_batch_size": "restart-only", "embed_cpu_threads": "restart-only",
                   "max_runtime_seq_tokens": "restart-only"}.get(k, "")
            print(f"{k:<12} {display:<20} {src:<16} {mut}")
        print()


def _domains_table(extra_dirs: Optional[List[str]] = None) -> None:
    if RICH_AVAILABLE and console:
        tbl = Table(title="[bold]Domain → Model[/bold]", show_lines=True)
        tbl.add_column("Domain", style="cyan")
        tbl.add_column("Model Path")
        for domain, name in DOMAIN_MODELS.items():
            path = resolve_model_path(name, extra_dirs)
            exists = "✓" if Path(path).exists() else "✗ MISSING"
            tbl.add_row(domain, f"[{'green' if '✓' in exists else 'red'}]{path} {exists}[/]")
        console.print(tbl)
    else:
        print(f"\n{'Domain':<12}  {'Model Path'}")
        print("─" * 60)
        for domain, path in DOMAIN_MODELS.items():
            exists = "✓" if Path(path).exists() else "✗ MISSING"
            print(f"{domain:<12}  {path}  {exists}")
        print()


def _dispatch(
    query: str,
    mgr: ModelManager,
    session_store: SessionStore,
    template_store: TemplateStore,
    settings: dict,
) -> None:
    """Classify and dispatch a query through the appropriate pipeline."""
    # Block here (briefly at most) until the background embedder thread is done.
    mgr.await_embedder(verbose=True)

    enhance = settings.get("enhance", True)
    kb_on = settings.get("kb", True)
    confirm_on = settings.get("confirm", True)
    render_mode = _normalize_render_mode(settings.get("render_mode", POLICY_V1_RENDER_DEFAULT))

    t0 = time.time()
    debug_log("dispatch.begin", query_chars=len(query or ""))

    def _is_followup_edit(text: str) -> bool:
        q = (text or "").strip().lower()
        return bool(re.match(r"^(make|modify|change|update|add|remove|tweak|improve|adjust|fix)\b", q))

    def _has_explicit_code_intent(text: str) -> bool:
        q = (text or "").lower()
        return bool(re.search(r"\b(python|javascript|typescript|function|class|method|algorithm|script|refactor|debug|compile|module)\b", q))

    def _parse_routing_correction(text: str) -> Optional[Tuple[str, List[str]]]:
        raw = (text or "").strip().lower()
        if not raw:
            return None
        tier_match = re.search(r"\btier\s*([123])\b", raw)
        if not tier_match:
            return None
        tier = f"TIER{tier_match.group(1)}"
        domains: List[str] = []
        for dom in ("code", "sql", "math", "medical", "legal", "web", "supervisor"):
            if re.search(r"(?<![a-zA-Z0-9])" + re.escape(dom) + r"(?![a-zA-Z0-9])", raw):
                domains.append(dom)
        if not domains:
            domains = ["supervisor"] if tier == "TIER1" else ["web"]
        if tier == "TIER3" and len(domains) < 2:
            tier = "TIER2"
        if tier == "TIER2" and len(domains) >= 2:
            tier = "TIER3"
        return tier, domains

    try:
        tier, domains, interpretation = classify_query(query, mgr, confirm_enabled=confirm_on)
        debug_log("dispatch.classified", tier=tier, domains=domains, interpretation=interpretation[:200])

        recent = session_store.get_recent_tasks(1) if session_store else []
        prev = recent[0] if recent else {}
        prev_domains = [str(d).lower() for d in (prev.get("domains", []) or []) if d]
        prev_tier = str(prev.get("tier", "TIER2")).upper()
        followup = _is_followup_edit(query)

        if followup and prev_domains:
            if tier == "TIER1":
                tier = "TIER2"
                domains = prev_domains
                cprint(f"[RoutingGuard] Follow-up continuity override → {tier} / {domains}", "dim")
                debug_log("dispatch.routing_override", reason="followup_from_tier1", tier=tier, domains=domains)
            elif prev_domains == ["web"] and domains == ["code"] and not _has_explicit_code_intent(query):
                tier = "TIER2"
                domains = ["web"]
                cprint("[RoutingGuard] Follow-up continuity override → TIER2 / ['web']", "dim")
                debug_log("dispatch.routing_override", reason="followup_web_over_code", tier=tier, domains=domains)
    except Exception as e:
        cprint(f"[ERROR] Classification failed: {e}", "red")
        debug_log("dispatch.classify_error", error=str(e))
        return

    if tier == "TIER1" or not confirm_on:
        # TIER1 is trivial — skip confirmation and route directly.
        cprint(f"[CoE] Routing: {tier} / {domains}", "dim")
    else:
        # TIER2 / TIER3 — show interpretation and allow correction.
        cprint(f"\n[CoE] I interpret this as: {interpretation}")
        cprint(f"  → Routing: {tier} / {domains}")
        try:
            correction = input("  [Enter to proceed, or type a correction] ").strip()
        except (EOFError, KeyboardInterrupt):
            correction = ""
        if correction:
            cprint(f"[CoE] Re-routing with correction: {correction}", "dim")
            parsed = _parse_routing_correction(correction)
            if parsed is not None:
                tier, domains = parsed
                interpretation = f"User routing override: {tier} / {domains}"
                cprint(f"[CoE] Applied routing override → {tier} / {domains}", "dim")
                debug_log("dispatch.routing_override", reason="user_correction", tier=tier, domains=domains)
            else:
                # Treat correction as query rewrite only when no explicit routing override is detected.
                query = correction
                try:
                    tier, domains, interpretation = classify_query(query, mgr, confirm_enabled=False)
                except Exception as e:
                    cprint(f"[ERROR] Re-classification failed: {e}", "red")
                    return

    try:
        domains = [d for d in domains if d in DOMAIN_MODELS] or ["supervisor"]
        if tier != "TIER1" and len(domains) >= 2:
            tier = "TIER3"
        elif tier == "TIER3" and len(domains) < 2:
            tier = "TIER2"

        if tier == "TIER1":
            debug_log("dispatch.route", tier="TIER1", domains=domains)
            output = run_tier1(query, mgr)
        elif tier == "TIER3":
            cprint("[CoE] This request spans multiple domains. Please split it into smaller prompts and send them one at a time.", "yellow")
            debug_log("dispatch.tier3_blocked", domains=domains)
            return
        else:
            domain = domains[0] if domains else "supervisor"
            debug_log("dispatch.route", tier="TIER2", domains=[domain])
            output = run_tier2(
                query, domain, mgr, session_store, template_store,
                enhance=enhance, kb_enabled=kb_on, render_mode=render_mode,
                resume_after_switch=False,
            )
    except Exception as e:
        cprint(f"[ERROR] Pipeline error: {e}", "red")
        debug_log("dispatch.pipeline_error", error=str(e), tier=tier, domains=domains)
        # Ensure specialist is unloaded on error
        if mgr.specialist_path():
            mgr.unload_specialist()
        return

    try:
        mgr.reconcile_residency(None, reason=f"post-{tier}")
    except Exception as audit_err:
        cprint(f"[Residency] audit warning: {audit_err}", "yellow")
        debug_log("dispatch.residency_warning", error=str(audit_err))

    if not (output or "").strip():
        cprint("[CoE] Empty pipeline output — returning friendly fallback message.", "yellow")
        debug_log("dispatch.empty_output", tier=tier, domains=domains)
        output = EMPTY_RESPONSE_MESSAGE

    elapsed = time.time() - t0

    try:
        render_response(output, title="Response")
    except Exception as print_err:
        # Fallback: plain print so the answer is never silently lost
        try:
            print("\n" + "─" * 60)
            print(output)
            print("─" * 60)
        except Exception:
            pass
        cprint(f"[WARN] Rich render failed ({print_err}); fell back to plain print.", "yellow")
        debug_log("dispatch.render_warning", error=str(print_err))

    debug_log("dispatch.end", tier=tier, domains=domains, elapsed_s=round(elapsed, 3), output_chars=len(output or ""))
    cprint(f"\n⏱  {elapsed:.1f}s  |  {tier} / {domains}", "dim")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="College of Experts — Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python demo.py\n"
            "  python demo.py --device cpu --no-kb\n"
            "  python demo.py --no-confirm --session myproject\n"
        ),
    )
    parser.add_argument("--device", default="dml",
                        choices=["dml", "cpu", "cuda"],
                        help="Compute device (default: dml — DirectML iGPU Radeon 890M)")
    parser.add_argument("--no-kb", action="store_true",
                        help="Disable KB retrieval")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable supervisor prompt enhancement (raw passthrough)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Disable interpretation confirmation prompt")
    parser.add_argument("--session", default=None,
                        help="Override session ID prefix")
    parser.add_argument("--embed-cpu-batch-size", type=int, default=None,
                        help="CPU embedding batch size for first-run template indexing")
    parser.add_argument("--embed-cpu-threads", type=int, default=None,
                        help="CPU threads for embedding model when running on CPU")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Enable JSONL breadcrumb trace at acceptance_runs/debug_trace.jsonl")
    args = parser.parse_args()

    debug_from_env = os.getenv("COE_DEBUG_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_enabled = bool(args.debug_trace or debug_from_env)
    set_debug_trace(debug_enabled)
    if debug_enabled:
        cprint(f"[DEBUG] Trace enabled: {_DEBUG_TRACE_PATH}", "dim")
        debug_log("process.start", argv=" ".join(sys.argv))

    # Load settings
    settings, sources = load_settings(args)

    # Startup banner
    enhance_str = "on" if settings["enhance"] else "off"
    confirm_str = "on" if settings["confirm"] else "off"
    kb_str      = "on" if settings["kb"] else "off"
    sess_str    = settings["session"] or "(auto)"

    banner_text = (
        f"[bold cyan]College of Experts — Interactive CLI v{VERSION}[/bold cyan]\n"
        f"device=[yellow]{settings['device']}[/yellow]  "
        f"enhance=[yellow]{enhance_str}[/yellow]  "
        f"confirm=[yellow]{confirm_str}[/yellow]  "
        f"kb=[yellow]{kb_str}[/yellow]  "
        f"session=[yellow]{sess_str}[/yellow]\n"
        f"Type [bold]/help[/bold] for commands."
        if RICH_AVAILABLE else
        f"College of Experts — Interactive CLI v{VERSION}\n"
        f"device={settings['device']}  enhance={enhance_str}  confirm={confirm_str}  kb={kb_str}\n"
        f"Type /help for commands."
    )
    if RICH_AVAILABLE and console:
        console.print(Panel(banner_text, border_style="cyan"))
    else:
        print(banner_text)

    # Instantiate core objects
    mgr = ModelManager(
        device=settings["device"],
        model_base_dir=settings.get("model_base_dir"),
        embed_cpu_batch_size=settings.get("embed_cpu_batch_size", 96),
        embed_cpu_threads=settings.get("embed_cpu_threads", 12),
        max_runtime_seq_tokens=settings.get("max_runtime_seq_tokens", 4096),
    )
    session_store = SessionStore(session_prefix=settings.get("session"))
    template_store = TemplateStore()

    try:
        mgr.startup()
    except FileNotFoundError as e:
        cprint(f"[FATAL] {e}", "red")
        sys.exit(1)
    except Exception as e:
        cprint(f"[FATAL] ModelManager startup failed: {e}", "red")
        sys.exit(1)

    # Init KB + template cache in background — waits for the embedder thread to finish,
    # then runs init_kb and template_store.startup so startup doesn't block the REPL.
    if EMBED_AVAILABLE:
        _kb_on = settings.get("kb", True)
        def _bg_init(_ts=template_store, _kb=_kb_on):
            emb = mgr.await_embedder(verbose=False)
            if emb:
                if _kb:
                    init_kb(emb)
                _ts.startup(emb)
                try:
                    emb.compact()
                except Exception:
                    pass
        threading.Thread(target=_bg_init, daemon=True).start()

    cprint(f"[SessionStore] Session: {session_store.session_id}", "dim")
    debug_log("session.started", session_id=session_store.session_id)
    _extra = [settings.get("model_base_dir")] if settings.get("model_base_dir") else None
    _domains_table(_extra)

    HELP_TEXT = """
[bold]Commands:[/bold]
  /quit, /exit       — Exit the shell
  /new               — Start a new session
  /status            — Show model and session status
  /memory [N]        — Show last N tasks from session (default 5)
  /domains           — Show domain → model table
  /enhance [on|off]  — Toggle auto prompt enhancement
  /confirm [on|off]  — Toggle interpretation confirmation
  /kb [on|off]       — Toggle KB retrieval
    /render [intermediate|final] — Toggle intermediate response panels
  /config            — Show all settings with source and mutability
  /config set KEY V  — Persist a mutable setting (writes config/demo_config.json)
  /help              — Show this message
  <anything else>    — Classify and dispatch as a query
""" if RICH_AVAILABLE else """
Commands:
  /quit /exit         Exit
  /new                New session
  /status             Model + session status
  /memory [N]         Last N tasks (default 5)
  /domains            Domain table
  /enhance [on|off]   Toggle enhancement
  /confirm [on|off]   Toggle confirmation
  /kb [on|off]        Toggle KB
    /render [intermediate|final] Toggle intermediate panels
  /config             Show all settings
  /config set K V     Persist a setting
  /help               This message
"""

    # Main loop
    while True:
        try:
            user_input = input("\n[CoE] > ").strip()
        except (EOFError, KeyboardInterrupt):
            cprint("\n[CoE] Goodbye.", "dim")
            mgr.shutdown()
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Shell commands ────────────────────────────────────────────────

        if cmd in ("/quit", "/exit"):
            cprint("[CoE] Goodbye. Shutting down models…", "dim")
            debug_log("shell.exit")
            mgr.shutdown()
            break

        elif cmd == "/new":
            session_store.clear_session()
            cprint(f"[CoE] New session: {session_store.session_id}")

        elif cmd == "/status":
            sv = mgr.supervisor_path()
            sp = mgr.specialist_path()
            stats = session_store.get_stats()
            cprint(f"  supervisor : {sv or '(not loaded)'}")
            cprint(f"  specialist : {sp or '(none)'}")
            cprint(f"  load_thread: {'active' if (mgr._load_thread and mgr._load_thread.is_alive()) else 'idle'}")
            cprint(f"  pending    : {mgr._pending_specialist_path or '(none)'}")
            cprint(f"  embedder   : {getattr(mgr._embedder, 'device', '(disabled)')}")
            cprint(f"  enhance    : {settings['enhance']}")
            cprint(f"  kb         : {settings['kb']}")
            cprint(f"  confirm    : {settings['confirm']}")
            cprint(f"  render     : {_normalize_render_mode(settings.get('render_mode'))}")
            cprint(f"  emb_batch  : {settings.get('embed_cpu_batch_size')}")
            cprint(f"  emb_threads: {settings.get('embed_cpu_threads')}")
            cprint(f"  seq_cap    : {settings.get('max_runtime_seq_tokens')}")
            cprint(f"  session    : {stats['session_id']}  tasks={stats['task_count']}")
            cprint(f"  artifacts  : {stats.get('artifact_count', 0)}")

        elif cmd.startswith("/memory"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            tasks = session_store.get_recent_tasks(n)
            if not tasks:
                cprint("[CoE] No tasks in current session.", "dim")
            else:
                for t in tasks:
                    tier = t.get("tier", "?")
                    doms = ",".join(t.get("domains", []))
                    q = t.get("query", t.get("query_snippet", ""))[:80]
                    cprint(f"  [{tier} | {doms}] {q}")

        elif cmd == "/domains":
            _domains_table([settings.get("model_base_dir")] if settings.get("model_base_dir") else None)

        elif cmd.startswith("/enhance"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["enhance"] = val
                sources["enhance"] = "runtime"
                save_setting("enhance", val)
                cprint(f"[CoE] enhance = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] enhance = {'on' if settings['enhance'] else 'off'}")

        elif cmd.startswith("/confirm"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["confirm"] = val
                sources["confirm"] = "runtime"
                save_setting("confirm", val)
                cprint(f"[CoE] confirm = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] confirm = {'on' if settings['confirm'] else 'off'}")

        elif cmd.startswith("/kb"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["kb"] = val
                sources["kb"] = "runtime"
                save_setting("kb", val)
                cprint(f"[CoE] kb = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] kb = {'on' if settings['kb'] else 'off'}")

        elif cmd.startswith("/render"):
            parts = user_input.split()
            if len(parts) > 1:
                mode_token = parts[1].lower()
                if mode_token in ("final", "final-only", "final_only"):
                    mode = "final only"
                elif mode_token in ("intermediate", "both", "all", "default"):
                    mode = "intermediate+final"
                else:
                    cprint("[CoE] render usage: /render intermediate|final", "yellow")
                    continue
                settings["render_mode"] = mode
                sources["render_mode"] = "runtime"
                save_setting("render_mode", mode)
                cprint(f"[CoE] render_mode = {mode}")
            else:
                cprint(f"[CoE] render_mode = {_normalize_render_mode(settings.get('render_mode'))}")

        elif cmd.startswith("/config set"):
            parts = user_input.split()
            if len(parts) < 4:
                cprint("[CoE] Usage: /config set KEY VALUE", "yellow")
            else:
                key = parts[2].lower()
                val_str = parts[3]
                if key == "device":
                    cprint("[CoE] 'device' is restart-only. Change in config/demo_config.json and restart.", "yellow")
                elif key in ("enhance", "confirm", "kb"):
                    val = val_str.lower() in ("on", "true", "1", "yes")
                    settings[key] = val
                    sources[key] = "runtime"
                    save_setting(key, val)
                    cprint(f"[CoE] {key} = {val}  (saved to config)")
                elif key == "render_mode":
                    mode = _normalize_render_mode(val_str)
                    settings["render_mode"] = mode
                    sources["render_mode"] = "runtime"
                    save_setting("render_mode", mode)
                    cprint(f"[CoE] render_mode = {mode}  (saved to config)")
                elif key in ("embed_cpu_batch_size", "embed_cpu_threads", "max_runtime_seq_tokens"):
                    try:
                        val = int(val_str)
                        if val < 1:
                            raise ValueError()
                    except ValueError:
                        cprint(f"[CoE] {key} must be a positive integer", "yellow")
                        continue
                    settings[key] = val
                    sources[key] = "runtime"
                    save_setting(key, val)
                    cprint(f"[CoE] {key} = {val}  (saved to config; restart to apply)")
                elif key == "session":
                    settings["session"] = val_str
                    sources["session"] = "runtime"
                    save_setting("session", val_str)
                    cprint(f"[CoE] session prefix = {val_str}  (saved to config)")
                else:
                    cprint(f"[CoE] Unknown setting '{key}'. Valid: device, enhance, confirm, kb, render_mode, embed_cpu_batch_size, embed_cpu_threads, max_runtime_seq_tokens, session", "yellow")

        elif cmd.startswith("/config"):
            _settings_table(settings, sources)

        elif cmd == "/help":
            if RICH_AVAILABLE and console:
                console.print(Markdown(HELP_TEXT))
            else:
                print(HELP_TEXT)

        # ── Query dispatch ────────────────────────────────────────────────
        else:
            try:
                _dispatch(user_input, mgr, session_store, template_store, settings)
            except Exception as dispatch_err:
                cprint(f"[ERROR] Unhandled exception in pipeline: {dispatch_err}", "red")
                debug_log("dispatch.unhandled_exception", error=str(dispatch_err))
                import traceback
                traceback.print_exc()
                # Ensure specialist is freed so the next query starts clean
                if mgr.specialist_path():
                    mgr.unload_specialist()


if __name__ == "__main__":
    main()
