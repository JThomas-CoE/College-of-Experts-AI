
import re
import json
import gc
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

# PHASE 1 — Core Inference Engine: ModelManager
# ═══════════════════════════════════════════════════════════════════════════

# V1 policy constants (documentation/alignment block)
# Keep aligned with ../ollamaDemo.md "V1 Policy Table (Source of Truth)".
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
THINK_BUDGET_UTIL =   64   # utility — classification, KB check, grading

# ── Output budgets — two distinct classes ────────────────────────────────
# ONNX/DML supervisor: DirectML pre-allocates a physical KV cache sized to
# max_length = input_tokens + think_budget + output_budget.
# Keeping these small is CRITICAL — the DML arena grows to the largest
# max_length ever used and never shrinks until the process exits.
# Rule of thumb: SUPERVISOR budget ≤ 2048 output tokens keeps Nanbeige 3B
# well under 10 GB including KV cache on Radeon 890M.
#
# Ollama specialist: output is just text in Python RAM — no GPU KV involved.
# These can remain large without any VRAM impact.
SUPERVISOR_OUTPUT_BUDGET_T1   =  768   # TIER1 supervisor — short concise answers
SUPERVISOR_OUTPUT_BUDGET_T2   = 2048   # TIER2 supervisor grading/synthesis pass
SUPERVISOR_OUTPUT_BUDGET_UTIL =  128   # utility calls — routing, grading verdicts

OUTPUT_BUDGET_T1         =  768    # TIER1  — kept aligned with supervisor cap
OUTPUT_BUDGET_T2         = 16384   # TIER2  — Ollama specialist primary + retry (RAM only)
OUTPUT_BUDGET_T3         = 16384   # TIER3  — Ollama specialist per-step (RAM only)
OUTPUT_BUDGET_SYNTHESIS  = 16384   # Ollama synthesis payload cap (RAM only)
SYNTHESIS_DRAFT_TOKEN_CAP = 8192   # safety cap for synthesis input draft payload
RETRY_DRAFT_TOKEN_CAP     = 1024   # cap prior-draft payload in retry prompts
OLLAMA_TIMEOUT_BASE_SEC   = 180    # fixed floor for local Ollama generation requests
OLLAMA_TIMEOUT_MAX_SEC    = 1800   # hard ceiling for very long generations
ENABLE_TIER2_SYNTHESIS = False     # Descoping synthesis for simpler demo
KEEP_SPECIALIST_WARM = True        # keep specialist resident across queries for stability/latency

EMPTY_RESPONSE_MESSAGE = (
    "Sorry — the models were not able to generate an acceptable response for this request. "
    "Please try rephrasing your prompt or splitting it into smaller steps."
)

# Folder-name substrings that identify thinking models (case-insensitive).
_THINKING_MODEL_MARKERS = ("nanbeige", "deepseek-r1", "qwq", "-r1")

def is_thinking_model(path: str) -> bool:
    """Return True if the model is known to emit <think>...</think> blocks."""
    p = path.lower()
    return any(m in p for m in _THINKING_MODEL_MARKERS)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models.
    If the block is unterminated (budget ran out mid-think), discard everything
    from <think> onward so callers never see raw reasoning tokens.
    """
    text = _THINK_BLOCK.sub("", text)
    open_idx = text.lower().find("<think>")
    if open_idx != -1:
        text = text[:open_idx]
    return text


def safe_first_line(text: str, default: str = "") -> str:
    """Return first non-empty stripped line; never raises on empty input."""
    s = (text or "").strip()
    if not s:
        return default
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return default


def normalize_grade_verdict(raw: str) -> Tuple[str, str]:
    """Return (verdict_word, verdict_line) from noisy grader output.

    verdict_word is one of: PASS, FAIL_INCOMPLETE, FAIL_OFFTOPIC, FAIL_FORMAT.
    If no canonical verdict is detected, treat it as a grading-format failure.
    """
    text = (raw or "").strip()
    first = safe_first_line(text, default="")

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


def clean_code_output(query: str, text: str) -> str:
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


def wants_code_only(query: str) -> bool:
    ql = query.lower()
    return any(k in ql for k in [
        "just give me", "only code", "no commentary", "no comments", "no docstring",
    ])


def normalize_python_function_output(query: str, text: str) -> str:
    """Stable code normalizer used before display/grading for code tasks."""
    return clean_code_output(query, text)


def detect_code_language(query: str) -> str:
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


def describe_code_language(query: str) -> str:
    language = detect_code_language(query)
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


def grade_code_output(query: str, draft: str) -> Tuple[str, str]:
    """Deterministic grader for code tasks.

    This avoids brittle LLM self-grading for code responses and ensures
    architecture-level coherence: generation is model-based, validation is rule-based.
    """
    cleaned = normalize_python_function_output(query, draft)
    if not cleaned:
        return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: empty output"

    language = detect_code_language(query)

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


def is_html_web_request(query: str) -> bool:
    q = (query or "").lower()
    web_markers = ["html", "css", "javascript", "web app", "webapp", "self contained", "self-contained"]
    return sum(1 for marker in web_markers if marker in q) >= 2


def grade_web_output(query: str, draft: str) -> Tuple[str, str]:
    text = (draft or "").strip().lower()
    if not text:
        return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: empty output"

    if is_html_web_request(query):
        has_html = any(tok in text for tok in ["<!doctype html", "<html", "<head", "<body"])
        has_behavior = any(tok in text for tok in ["<script", "addEventListener", "setinterval", "settimeout", "function "])
        if not has_html:
            return "FAIL_FORMAT", "FAIL_FORMAT: missing HTML document structure"
        if not has_behavior:
            return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: missing timer behavior/javascript"

    return "PASS", "PASS"


def deterministic_grade_guard(query: str, draft: str, domain: str) -> Optional[Tuple[str, str]]:
    """Return an obvious deterministic failure, else None.

    This is a confirmation guard only. It catches empty, scaffolded, or visibly
    malformed outputs before any model-based grading is trusted.
    """
    text = (draft or "").strip()
    if not text:
        return "FAIL_INCOMPLETE", "FAIL_INCOMPLETE: empty output"

    if domain == "code":
        word, line = grade_code_output(query, draft)
        if word != "PASS":
            return word, line
        return None

    if domain == "web":
        web_word, web_line = grade_web_output(query, draft)
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


def grade_with_specialist_self_check(
    query: str,
    draft: str,
    domain: str,
    mgr: "ModelManager",
    skill_guidance: str = None,
    template_scaffold: str = None,
) -> Tuple[str, str]:
    """Grade with the currently loaded specialist only, plus deterministic confirmation.
    
    The grader runs as a completely separate API call with its own fresh context
    and grader persona at near-zero temperature. The failed draft is truncated to
    query + draft text as plain strings — it is never fed back into the retry.
    The failure reason is formatted as a second-person actionable clause for
    direct injection into the retry prompt template.
    """
    deterministic_fail = deterministic_grade_guard(query, draft, domain)
    if deterministic_fail is not None:
        return deterministic_fail

    code_language = describe_code_language(query) if domain == "code" else None
    grader_role = f"strict {code_language} code evaluator" if code_language else f"strict {domain} specialist evaluator"
    language_note = (
        f" The candidate must be evaluated as {code_language} code."
        if code_language else ""
    )

    # Specialist grader (Ollama, 32K context): pass the FULL draft.
    # The dynamic num_ctx in _generate_via_ollama sizes the context window from
    # the actual prompt length, so no hardcoded truncation is needed here.
    grade_query = (query or "")[:600]   # query is always short; 600-char cap is generous
    grade_draft = (draft or "")         # full draft — no truncation

    grade_system = (
        f"You are a {grader_role}. Your job is to identify whether the candidate response "
        f"correctly satisfies the user's request.{language_note} "
        "If it fails, state the failure reason as a direct second-person verb clause "
        "(e.g. 'did not apply the @njit decorator', 'used plain Python loops instead of vectorised operations', "
        "'returned HTML instead of JSON'). "
        "Be specific and concise — one to two sentences maximum. "
        "Output exactly one plain-text verdict line. No preamble. No bullets. No code fences."
    )
    if skill_guidance:
        grade_system += (
            f"\n\nEvaluation criteria — guidelines that were applied during generation "
            f"(check the response against these requirements):\n{skill_guidance}"
        )
    if template_scaffold:
        grade_system += (
            f"\n\nRequired output structure — the response must conform to this format:\n{template_scaffold}"
        )
    grade_prompt = (
        f"User prompt:\n{grade_query}\n\n"
        f"Candidate response:\n{grade_draft}\n\n"
        "Evaluate whether the candidate response is accurate, complete, and on-topic. "
        f"Apply the grading rules carefully.{language_note} "
        "Output exactly one line and nothing else, starting with one of:\n"
        "PASS\n"
        "FAIL_INCOMPLETE: <direct second-person failure clause>\n"
        "FAIL_OFFTOPIC: <direct second-person failure clause>\n"
        "FAIL_FORMAT: <direct second-person failure clause>"
    )
    verdict = mgr.generate_specialist_grade(
        grade_prompt,
        system=grade_system,
        max_tokens=128,
    )
    verdict_word, verdict_line = normalize_grade_verdict(verdict)

    # Deterministic guard runs after LLM grade too (belt-and-suspenders)
    deterministic_fail = deterministic_grade_guard(query, draft, domain)
    if deterministic_fail is not None:
        return deterministic_fail

    if verdict_line == "FAIL_FORMAT: non-canonical grader output":
        repair_prompt = (
            f"User prompt:\n{grade_query}\n\n"
            f"Candidate response:\n{grade_draft}\n\n"
            "Your previous grader output was invalid. Re-grade now. "
            "Return exactly one line only in one of these forms:\n"
            "PASS\n"
            "FAIL_INCOMPLETE: <direct second-person failure clause>\n"
            "FAIL_OFFTOPIC: <direct second-person failure clause>\n"
            "FAIL_FORMAT: <direct second-person failure clause>"
        )
        retry_verdict = mgr.generate_specialist_grade(
            repair_prompt,
            system=grade_system,
            max_tokens=128,
        )
        retry_word, retry_line = normalize_grade_verdict(retry_verdict)
        if retry_line == "FAIL_FORMAT: non-canonical grader output":
            return "FAIL_FORMAT", "FAIL_FORMAT: specialist self-grader produced non-canonical verdict"
        return retry_word, retry_line
    return verdict_word, verdict_line


def grade_candidate(
    query: str,
    draft: str,
    domain: str,
    mgr: "ModelManager",
    grade_system: str,
    self_grade: bool,
    use_specialist_self_grade: bool,
    skill_guidance: str = None,
    template_scaffold: str = None,
) -> Tuple[str, str]:
    if not self_grade:
        return "PASS", "PASS: self-grading disabled by user"
    if use_specialist_self_grade and domain in SPECIALIST_SELF_GRADE_DOMAINS:
        return grade_with_specialist_self_check(
            query, draft, domain, mgr,
            skill_guidance=skill_guidance,
            template_scaffold=template_scaffold,
        )
    return grade_output(query, draft, domain, mgr, grade_system, skill_guidance=skill_guidance, template_scaffold=template_scaffold)


def grade_output(
    query: str,
    draft: str,
    domain: str,
    mgr: "ModelManager",
    grade_system: str,
    skill_guidance: str = None,
    template_scaffold: str = None,
) -> Tuple[str, str]:
    """Policy-based grading router.

    - all domains: deterministic hard-fail guard first
    - then: supervisor grading + canonical verdict normalization

    The ONNX supervisor has a ~4096-token context limit. For large drafts we use
    a head-60% / tail-40% excerpt so the grader sees both structure (start) and
    completion quality (end) rather than a raw front-only truncation.
    """
    deterministic_fail = deterministic_grade_guard(query, draft, domain)
    if deterministic_fail is not None:
        return deterministic_fail

    language_note = ""
    if domain == "code":
        language_note = f" Evaluate code using {describe_code_language(query)} conventions."

    # Smart head+tail truncation for the context-limited ONNX supervisor grader.
    # Budget: 4096 ctx – ~400 prompt overhead – 128 output = ~3568 tokens ≈ 10700 chars.
    _SUPERVISOR_GRADE_CHAR_CAP = 10000
    draft_for_grade = draft or ""
    if len(draft_for_grade) > _SUPERVISOR_GRADE_CHAR_CAP:
        head_chars = int(_SUPERVISOR_GRADE_CHAR_CAP * 0.6)
        tail_chars = _SUPERVISOR_GRADE_CHAR_CAP - head_chars
        draft_for_grade = (
            draft_for_grade[:head_chars]
            + f"\n\n...[{len(draft_for_grade) - _SUPERVISOR_GRADE_CHAR_CAP} chars omitted]...\n\n"
            + draft_for_grade[-tail_chars:]
        )

    skill_criteria_block = ""
    if skill_guidance:
        skill_criteria_block = (
            f"Guidelines applied during generation (use as evaluation criteria):\n{skill_guidance}\n\n"
        )
    # Supervisor path: cap scaffold at 600 chars to stay within ~10K char grader budget.
    _SCAFFOLD_CHAR_CAP = 600
    template_block = ""
    if template_scaffold:
        capped = template_scaffold[:_SCAFFOLD_CHAR_CAP]
        if len(template_scaffold) > _SCAFFOLD_CHAR_CAP:
            capped += f"\n...[{len(template_scaffold) - _SCAFFOLD_CHAR_CAP} chars omitted]"
        template_block = f"Required output structure (check format compliance):\n{capped}\n\n"
    grade_prompt = (
        f"Task: {query}\nOutput: {draft_for_grade}\n\n"
        f"{skill_criteria_block}"
        f"{template_block}"
        f"Assess the output. Reply with exactly one line, starting with one of:\n"
        f"PASS\n"
        f"FAIL_INCOMPLETE: <one line reason>\n"
        f"FAIL_OFFTOPIC: <one line reason>\n"
        f"FAIL_FORMAT: <one line reason>{language_note}"
    )
    verdict = mgr.generate_supervisor(
        grade_prompt,
        system=grade_system,
        max_tokens=SUPERVISOR_OUTPUT_BUDGET_UTIL,
        temperature=0.01,
        think_budget=THINK_BUDGET_UTIL,
    )
    return normalize_grade_verdict(verdict)


_VERDICT_QUALITY = {
    "PASS": 3,
    "FAIL_FORMAT": 2,
    "FAIL_INCOMPLETE": 1,
    "FAIL_OFFTOPIC": 0,
}


def candidate_quality_score(verdict_word: str, text: str) -> float:
    """Return a comparable quality score for benchmark-style candidate selection."""
    base = _VERDICT_QUALITY.get((verdict_word or "").upper(), 0)
    length_bonus = min(len((text or "").strip()), 400) / 4000.0
    return base + length_bonus


def pick_better_candidate(
    left_text: str,
    left_verdict: str,
    left_line: str,
    right_text: str,
    right_verdict: str,
    right_line: str,
) -> Tuple[str, str, str]:
    """Choose better response using verdict quality first, verbosity second."""
    left_score = candidate_quality_score(left_verdict, left_text)
    right_score = candidate_quality_score(right_verdict, right_text)

    if right_score > left_score:
        return right_text, right_verdict, right_line
    if right_score < left_score:
        return left_text, left_verdict, left_line

    if len((right_text or "").strip()) > len((left_text or "").strip()):
        return right_text, right_verdict, right_line
    return left_text, left_verdict, left_line


def should_retry_tier2(verdict_word: str, text: str) -> bool:
    """Benchmark-style retry gate: retry once on weak/failed specialist draft."""
    if verdict_word in FAIL_CATEGORIES:
        return True
    return len((text or "").strip()) < 20


def should_trigger_supervisor_fallback(verdict_word: str, domain: str) -> bool:
    """Benchmark-style fallback gate after primary pipeline selection."""
    if domain == "supervisor":
        return False
    # Specialist domains use fine-tuned MoE models; the supervisor (Nanbeige) is a
    # general-purpose model and is NOT a better fallback for code or web tasks.
    # Sending a failed code/web response to the supervisor wastes a full inference
    # budget and typically produces worse output than the original specialist.
    if domain in {"code", "web"}:
        return False
    return verdict_word in FAIL_CATEGORIES


def run_supervisor_fallback(
    query: str,
    mgr: "ModelManager",
    failure_summary: str,
    max_tokens: int,
    think_budget: int,
    on_token: Optional[callable] = None,
) -> str:
    """Generate a fresh fallback answer from the original task and failure summary only."""
    fb_prompt = (
        "Answer this question carefully and accurately from scratch.\n\n"
        f"QUESTION: {query}\n\n"
        f"Failure to avoid: {failure_summary or 'previous answer was incomplete or malformed'}\n\n"
        "Do not continue, repair, or quote the failed draft. Generate a new answer that directly satisfies the original question.\n\n"
        "Think through this step by step, then provide your final answer:"
    )
    return generate_supervisor_answer_resilient(
        mgr=mgr,
        prompt=fb_prompt,
        system="You are an advanced reasoning supervisor.",
        max_tokens=max_tokens,
        temperature=0.2,
        think_budget=think_budget,
        retry_label="fallback",
        on_token=on_token,
    )

def classify_followup(
    query: str,
    prior_query: str,
    mgr: "ModelManager",
) -> str:
    """Supervisor-arbitrated follow-up detection.

    Returns one of: "FOLLOWUP" | "NEW_TASK" | "AMBIGUOUS"

    FOLLOWUP  → inject prior artifact into specialist context
    NEW_TASK  → clean stateless call, no prior context
    AMBIGUOUS → pipeline should ask user for clarification before proceeding
    """
    if not prior_query or not mgr._supervisor_handle:
        return "NEW_TASK"
    prompt = (
        f"Previous task: \"{prior_query.strip()[:200]}\"\n"
        f"Current request: \"{query.strip()[:200]}\"\n\n"
        "Classify the current request:\n"
        "- FOLLOWUP: the request fixes, modifies, or references the previous task's output "
        "(look for: 'fix', 'change', 'the [noun]', 'it', 'they', 'this', 'that', 'broken', "
        "'not working', 'update', 'add to', definite-article references without a new subject)\n"
        "- NEW_TASK: the request is a completely new, independent task with its own full specification "
        "(look for: 'make me a', 'create a', 'build a', 'write a', 'generate a', new topic noun)\n"
        "- AMBIGUOUS: you cannot confidently determine which it is\n\n"
        "Reply with exactly one word: FOLLOWUP, NEW_TASK, or AMBIGUOUS"
    )
    system = (
        "You are a precise query classifier. "
        "A short request referencing 'the buttons', 'the function', 'the app', 'it', 'they', "
        "or any definite-article noun without introducing a new subject is almost always FOLLOWUP. "
        "Only output NEW_TASK if the request clearly introduces a brand-new specification. "
        "Output AMBIGUOUS only if genuinely uncertain. "
        "Output exactly one word: FOLLOWUP, NEW_TASK, or AMBIGUOUS."
    )
    try:
        verdict = mgr.generate_supervisor(
            prompt,
            system=system,
            max_tokens=SUPERVISOR_OUTPUT_BUDGET_UTIL,
            temperature=0.01,
            think_budget=0,
            disable_thinking=True,
        ).strip().upper()
    except Exception:
        return "AMBIGUOUS"

    if "FOLLOWUP" in verdict:
        result = "FOLLOWUP"
    elif "NEW_TASK" in verdict or "NEW" in verdict:
        result = "NEW_TASK"
    else:
        result = "AMBIGUOUS"

    debug_log("classify_followup", prior=prior_query[:60], query=query[:60], verdict=verdict, result=result)
    return result


def generate_supervisor_answer_resilient(
    mgr: "ModelManager",
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    think_budget: int,
    retry_label: str = "answer",
    on_token: Optional[callable] = None,
) -> str:
    """Generate with supervisor and retry once without thinking if empty."""
    result = mgr.generate_supervisor(
        prompt,
        system=system,
        max_tokens=min(max_tokens, SUPERVISOR_OUTPUT_BUDGET_T2),
        temperature=temperature,
        think_budget=think_budget,
        on_token=on_token,
    )
    if (result or "").strip():
        return result

    debug_log(f"supervisor.{retry_label}.empty_primary")
    retry_system = (
        f"{system}\n\n"
        "Answer directly and concretely. Do not spend time on hidden reasoning. "
        "If uncertain, say so briefly and then give the best grounded answer you can."
    ).strip()
    # Retry is a recovery pass — not streamed (user already saw the header)
    retry_result = mgr.generate_supervisor(
        prompt,
        system=retry_system,
        max_tokens=min(max_tokens, SUPERVISOR_OUTPUT_BUDGET_T2),
        temperature=max(temperature, 0.1),
        think_budget=0,
    )
    if (retry_result or "").strip():
        debug_log(f"supervisor.{retry_label}.retry_recovered", output_chars=len(retry_result))
        return retry_result
    debug_log(f"supervisor.{retry_label}.retry_empty")
    return result


def build_retry_prompt_from_template(
    original_query: str,
    fail_reason: str,
    kb_snippet: Optional[str] = None,
) -> str:
    """Build retry prompt from a pure Python template — no LLM call, no failed draft re-injection.

    Architecture contract:
    - The failed draft lives only as a Python string in RAM for candidate comparison.
    - It is NEVER sent back to the Ollama API.
    - KB snippet (if any) is retrieved using the grader's fail reason as the search key,
      not the original query, for higher precision retrieval.
    - The retry prompt is strictly: original_query + failure directive + optional KB.
    - The specialist sees this as a fresh first-attempt, with a focused correction hint.
    
    Template format (Option B — requirement framing, not failure framing):
        [original user query]

        Important: [fail_reason — as second-person verb clause from grader].
        [Consider also: kb_snippet — only if retrieved]
    """
    parts = [original_query.strip()]
    parts.append(f"Important: you {fail_reason.lstrip('you ').strip()}.")
    if kb_snippet and kb_snippet.strip():
        parts.append(f"Consider also: {kb_snippet.strip()}")
    return "\n\n".join(parts)


def build_retry_guidance(
    query: str,
    task_prompt: str,
    fail_category: str,
    fail_reason: str,
    mgr: "ModelManager",
    kb_context: Optional[str] = None,
) -> str:
    """Legacy shim — returns empty string. Retry prompt now built by build_retry_prompt_from_template."""
    return ""


def build_fresh_retry_prompt(
    original_task_prompt: str,
    fail_category: str,
    fail_reason: str,
    retry_guidance: str,
    kb_context: Optional[str] = None,
) -> str:
    """Legacy shim kept for callsite compatibility — delegates to the template builder."""
    # Extract the original user query from the task prompt (first line / up to first double newline)
    original_query = original_task_prompt.split("\n\n")[0].strip() if original_task_prompt else ""
    return build_retry_prompt_from_template(
        original_query=original_query,
        fail_reason=fail_reason,
        kb_snippet=kb_context,
    )


def format_prompt(model_path: str, prompt: str, system: str = "") -> str:
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


def estimate_token_count(text: str) -> int:
    """Cheap token estimate used when tokenizers live only inside workers."""
    s = text or ""
    if not s:
        return 0
    return max(1, (len(s) + 3) // 4)


def trim_text_to_estimated_tokens(text: str, max_tokens: int) -> str:
    """Trim text conservatively using a rough chars-per-token estimate."""
    if max_tokens <= 0:
        return ""
    s = text or ""
    if estimate_token_count(s) <= max_tokens:
        return s
    char_budget = max_tokens * 4
    return s[:char_budget]


def ollama_request_timeout_seconds(max_tokens: int) -> int:
    """Return a conservative HTTP timeout for local Ollama generations."""
    scaled = OLLAMA_TIMEOUT_BASE_SEC + int(max(0, max_tokens) * 0.10)
    return max(OLLAMA_TIMEOUT_BASE_SEC, min(OLLAMA_TIMEOUT_MAX_SEC, scaled))


def fit_draft(draft: str, prompt_skeleton: str, mgr: "ModelManager",
               output_budget: int, think_budget: int, system: str = "",
               max_draft_tokens: Optional[int] = None) -> str:
    """Trim draft to fit the supervisor context budget conservatively."""
    if not draft:
        return draft
    if not mgr.supervisor_path():
        return draft
    ctx = mgr.supervisor_ctx_limit()
    formatted = format_prompt(mgr.supervisor_path(), prompt_skeleton, system)
    skeleton_toks = estimate_token_count(formatted)
    reserve = output_budget + think_budget + 32
    available = max(ctx - skeleton_toks - reserve, 0)
    if max_draft_tokens is not None:
        available = min(available, max_draft_tokens)
    if available == 0:
        return ""
    if estimate_token_count(draft) <= available:
        return draft
    return trim_text_to_estimated_tokens(draft, available)


@dataclass
class ModelHandle:
    path: str
    model: object       # og.Model
    tokenizer: object   # og.Tokenizer


class ModelManager:
    """
    Hybrid Model Lifecycle Manager.

    Persistent pool (loaded natively in RAM/VRAM):
        _supervisor  — Nanbeige INT4 supervisor (ONNX/DML)
        _embedder    — BGE-M3 embedding model (sentence-transformers)

    Swappable pool (API calls to Ollama):
        _specialist  — Hits local Ollama server, maintaining 'keep_alive'
    """

    def __init__(
        self,
        device: str = "dml",
        model_base_dir: str = None,
        embed_cpu_batch_size: int = 96,
        embed_cpu_threads: int = 12,
        max_runtime_seq_tokens: int = 16384,
    ):
        self.device = device
        self._extra_dirs: List[str] = [model_base_dir] if model_base_dir else []
        self._embed_cpu_batch_size = max(8, int(embed_cpu_batch_size))
        self._embed_cpu_threads = max(1, int(embed_cpu_threads))
        self._max_runtime_seq_tokens = max(1024, int(max_runtime_seq_tokens))
        
        self._supervisor_path: Optional[str] = None
        self._supervisor_handle: Optional[ModelHandle] = None
        
        self._embedder: Optional["EmbeddingManager"] = None
        self._specialist_path: Optional[str] = None
        self._last_specialist_num_ctx: Optional[int] = None   # reused by grader + retry
        self._load_error: Optional[str] = None

    def _resolve(self, name: str) -> str:
        if name.startswith("ollama://"):
            return name
        return resolve_model_path(name, self._extra_dirs)

    def _load_supervisor_handle(self) -> None:
        import onnxruntime_genai as og
        sv_path = self._resolve(DOMAIN_MODELS["supervisor"])
        if not Path(sv_path).exists():
            raise FileNotFoundError(f"Supervisor model not found: {sv_path}")

        cprint(f"[ModelManager] Loading Supervisor natively on {self.device}: {sv_path}")
        t0 = time.time()
        self._supervisor_path = sv_path
        
        # Load the actual model inside the main process Memory Space
        model = og.Model(sv_path)
        tokenizer = og.Tokenizer(model)
        self._supervisor_handle = ModelHandle(
            path=sv_path,
            model=model,
            tokenizer=tokenizer
        )
        cprint(f"[ModelManager] Supervisor ready ({time.time() - t0:.1f}s)", "green")

    def startup(self) -> None:
        # Ensure Ollama daemon is running before loading anything
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/", timeout=1)
        except Exception:
            cprint("[ModelManager] Waking up Ollama daemon in the background...", "yellow")
            import subprocess
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            for _ in range(15):
                try:
                    time.sleep(1)
                    urllib.request.urlopen("http://localhost:11434/", timeout=1)
                    break
                except Exception:
                    pass
        
        self._load_supervisor_handle()
        if EMBED_AVAILABLE:
            embed_device = "cuda" if self.device == "cuda" else "cpu"
            cprint(f"[ModelManager] Loading embedder: {EMBEDDING_MODEL} on {embed_device}")
            t0 = time.time()
            self._embedder = EmbeddingManager(
                model_name=EMBEDDING_MODEL,
                device=embed_device,
                cpu_batch_size=self._embed_cpu_batch_size,
                cpu_threads=self._embed_cpu_threads,
            )
            cprint(f"[ModelManager] Embedder ready ({time.time() - t0:.1f}s)", "green")

    def shutdown(self) -> None:
        if self._specialist_path and self._specialist_path.startswith("ollama://"):
            model_name = self._specialist_path.replace("ollama://", "")
            try:
                import urllib.request
                import json
                payload = {"model": model_name, "keep_alive": 0}
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=45)
            except Exception:
                pass

        self.unload_specialist()
        if self._supervisor_handle:
            del self._supervisor_handle.tokenizer
            del self._supervisor_handle.model
            self._supervisor_handle = None
        self._supervisor_path = None
        self._embedder = None
        gc.collect()

        try:
            import os
            if os.name == 'nt':
                os.system("taskkill /f /im ollama.exe >nul 2>&1")
                os.system("taskkill /f /im ollama_llama_server.exe >nul 2>&1")
            else:
                os.system("pkill ollama >/dev/null 2>&1")
        except Exception:
            pass

    def reset_to_supervisor_only(self, reason: str = "") -> None:
        self.unload_specialist()
        if self._embedder and hasattr(self._embedder, "compact"):
            try:
                self._embedder.compact()
            except Exception:
                pass
        gc.collect()

    def ensure_specialist_for_model(self, model_path: str) -> str:
        if not model_path.startswith("ollama://"):
            return "skip"
        if self._specialist_path != model_path:
            self.unload_specialist(force_evict=True)
            self._specialist_path = model_path
            return "switch"
        return "reuse"

    def await_specialist(self) -> None:
        pass

    def unload_specialist(self, force_evict: bool = False) -> None:
        """Clear internal tracking. If force_evict is True, explicitly tell Ollama to unload the model from VRAM.

        We no longer send keep_alive:0 by default because:
        - The same model is reused as both generator and grader (different system prompt only)
        - Explicit eviction causes a full model reload for the next call (~30-60s on 40B)
        - Ollama will evict naturally when a different model is requested or the daemon exits
        HOWEVER: When explicitly switching domains, force_evict must be True to prevent stacking VRAM allocations.
        """
        if self._specialist_path and self._specialist_path.startswith("ollama://"):
            model_name = self._specialist_path.replace("ollama://", "")
            if force_evict:
                cprint(
                    f"[ModelManager] Evicting specialist from VRAM: {model_name}",
                    "dim",
                )
                try:
                    import urllib.request
                    import json
                    payload = {"model": model_name, "keep_alive": 0}
                    req = urllib.request.Request(
                        "http://localhost:11434/api/generate",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(req, timeout=45)
                except Exception as e:
                    debug_log("ollama.evict_error", model=model_name, error=str(e))
            else:
                cprint(
                    f"[ModelManager] Clearing specialist tracking "
                    f"(Ollama model stays resident): {model_name}",
                    "dim",
                )
        self._specialist_path = None
        self._last_specialist_num_ctx = None

    def specialist_path(self) -> Optional[str]:
        return self._specialist_path

    def reconcile_residency(self, expected_domain: Optional[str], reason: str = "") -> None:
        pass

    def supervisor_path(self) -> str:
        return self._supervisor_path or ""

    def _generate_via_ollama(
        self,
        model_path: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        on_token: Optional[callable] = None,
        loop_guard: bool = True,
    ) -> Tuple[str, float]:
        import urllib.request
        model_name = model_path.replace("ollama://", "")
        streaming = on_token is not None
        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system,
            "stream": streaming,
            "keep_alive": -1,  # Keep hot in memory
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        # ── Static num_ctx — preallocate to avoid model reloading ─────────────
        # CRITICAL: Ollama treats ANY change to `num_ctx` as a KV cache invalidation
        # and forcefully evicts and RELOADS the entire model weights into VRAM.
        # Starting small (4096) and growing dynamically on follow-ups (~8000+)
        # guarantees a horribly slow 10-20s model reload during a conversation.
        # Solution: Statically allocate the max configured session context size upfront.
        _input_token_estimate = (len(system) + len(prompt)) // 3
        _model_ctx_cap = (
            32768 if ("CoE-python2" in model_path or "CoE-WEB2" in model_path) else 16384
        )
        
        # Statically allocate the configured maximum (e.g., 16384) clamped by model limits
        _num_ctx = min(_model_ctx_cap, self._max_runtime_seq_tokens)
        
        # Track the active context size solely to catch regressions
        _is_active_specialist = (model_path == self._specialist_path)
        if _is_active_specialist:
            self._last_specialist_num_ctx = _num_ctx

        payload["options"]["num_ctx"] = _num_ctx
        debug_log(
            "ollama.num_ctx",
            model=model_name,
            est_input_tok=_input_token_estimate,
            num_ctx=_num_ctx,
            max_tokens=max_tokens,
        )

        # ── MoE model overrides ───────────────────────────────────────────────
        if "CoE-python2" in model_path or "CoE-WEB2" in model_path:
            if "CoE-python2" in model_path:
                payload["options"]["temperature"] = 0.9
            elif "CoE-WEB2" in model_path:
                payload["options"]["temperature"] = 0.2
            
            # Disable repeat penalty completely. Any penalty > 1.0 suppresses the Qwen end-of-turn 
            # and end-of-file tokens and forces the model to structurally repeat markdown blocks.
            payload["options"]["top_p"] = 0.90
            payload["options"]["repeat_penalty"] = 1.0
            
            # Explicitly enforce sequence limits to prevent endless markdown structural loops
            payload["options"]["stop"] = ["<|im_end|>", "<|endoftext|>"]

            # Inform Ollama's MoE runtime exactly how many experts to activate.
            # Both CoE-python2 and CoE-WEB2 are 40B-A3B models with 64 total experts
            # and 11 active per token.  Without these hints Ollama may default to
            # the architecture's "expert_used_count" GGUF field which is absent on
            # older quants, leading to silent single-expert inference.
            payload["options"]["num_experts_used"] = 11
            payload["options"]["num_experts_per_tok"] = 11

        t0 = time.time()
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            request_timeout = ollama_request_timeout_seconds(max_tokens)
            resp = urllib.request.urlopen(req, timeout=request_timeout)

            if streaming:
                # NDJSON streaming: each line is a JSON object with a 'response' chunk
                chunks: list = []
                
                # --- Streaming Loop Killer (N-Gram Detector) ---
                # Tracks the last N tokens to brutally sever the socket connection if 
                # the model starts infinitely outputting the exact same macro-structures
                history_buffer = []
                MAX_BUFFER = 60
                loop_detected = False
                
                for raw_line in resp:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        chunk_data = json.loads(raw_line.decode("utf-8"))
                    except Exception:
                        continue
                    token = chunk_data.get("response", "")
                    
                    if token:
                        history_buffer.append(token)
                        if len(history_buffer) > MAX_BUFFER:
                            history_buffer.pop(0)

                        chunks.append(token)
                        on_token(token)
                        
                        # --- Evaluate loop heuristics every 10 tokens ---
                        if loop_guard and len(history_buffer) >= 30 and len(chunks) % 10 == 0:
                            # Heuristic 1: Micro-stutter (short math/word loops)
                            # e.g., 5 tokens repeated 6 times (30 tokens)
                            # Guard: only fire if the repeating pattern contains at least one
                            # identifier of 4+ chars. Patterns made entirely of single-char
                            # literals, brackets, quotes and commas are structured data
                            # (chess boards, coordinate arrays, lookup tables) — not loops.
                            for size in range(3, 8):
                                pattern = history_buffer[-size:]
                                matched_reps = 0
                                for i in range(1, 6):
                                    start_idx = len(history_buffer) - (size * (i + 1))
                                    if start_idx < 0: break
                                    compare_slice = history_buffer[start_idx : start_idx + size]
                                    if compare_slice == pattern:
                                        matched_reps += 1
                                    else:
                                        break
                                if matched_reps >= 5 and re.search(r'\w{4,}', ''.join(pattern)):
                                    loop_detected = True
                                    break
                            
                            # Heuristic 2: Macro-loop (long structural HTML/code repetition)
                            # e.g., 15-30 tokens repeated exactly 2+ times
                            # Same data-literal guard: real loops always contain word tokens
                            # of 4+ chars; board rows / numeric arrays / symbol tables don't.
                            if not loop_detected:
                                for size in range(15, 30):
                                    if len(history_buffer) < size * 2: continue
                                    pattern = history_buffer[-size:]
                                    compare_slice = history_buffer[-(size*2):-size]
                                    if pattern == compare_slice and re.search(r'\w{4,}', ''.join(pattern)):
                                        loop_detected = True
                                        break
                        
                        if loop_detected:
                            cprint("\n[!] FATAL LOOP DETECTED. SEVERING STREAM.", "red")
                            debug_log("ollama.stream_severed", reason="ngram_repetition")
                            
                            # Clean up broken markdown if we severed inside a codeblock
                            joined_chunks = "".join(chunks)
                            if joined_chunks.count("```") % 2 != 0:
                                cleanup_token = "\n```\n"
                                chunks.append(cleanup_token)
                                on_token(cleanup_token)
                            
                            break  # Kill the socket parsing loop immediately

                    if chunk_data.get("done", False):
                        break
                elapsed = time.time() - t0
                return "".join(chunks), elapsed
            else:
                data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - t0
                return data.get("response", ""), elapsed
        except Exception as e:
            raise RuntimeError(f"Ollama API generation failed: {e}")



    def _generate_with(
        self,
        handle: ModelHandle,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        think_budget: int = THINK_BUDGET_T2,
        disable_thinking: bool = False,
        on_token: Optional[callable] = None,
    ) -> str:
        import onnxruntime_genai as og

        formatted = format_prompt(handle.path, prompt, system)
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

        if is_thinking_model(handle.path) and not disable_thinking:
            # Hard safety invariant: never create a generator when the
            # formatted input already fills (or exceeds) model context.
            initial_headroom = effective_ctx_limit - len(input_tokens) - 1
            if initial_headroom <= 0:
                gc.collect()
                return ""

            # ── Phase 1: thinking ────────────────────────────────────────
            reserve_for_answer = min(max_tokens, max(256, min(1024, initial_headroom // 3)))
            think_limit = min(think_budget, max(initial_headroom - reserve_for_answer - 1, 0))
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
                try:
                    g1.append_tokens(input_tokens)
                    think_tokens: list = []
                    found_close = False
                    while not g1.is_done() and len(think_tokens) < think_limit:
                        g1.generate_next_token()
                        think_tokens.append(g1.get_next_tokens()[0])
                        if len(think_tokens) % 4 == 0:
                            if "</think>" in handle.tokenizer.decode(think_tokens).lower():
                                found_close = True
                                break
                    think_text = handle.tokenizer.decode(think_tokens)
                finally:
                    del g1
                    del p1
                    gc.collect()

            if not found_close:
                think_text = think_text.rstrip() + "\n</think>\n"

            # ── Phase 2: answer ──────────────────────────────────────────
            phase2_text = formatted + think_text
            phase2_tokens = handle.tokenizer.encode(phase2_text)
            if len(phase2_tokens) > effective_ctx_limit - 32:
                phase2_tokens = phase2_tokens[-(effective_ctx_limit - 32):]
            answer_budget = min(max_tokens, effective_ctx_limit - len(phase2_tokens) - 10)
            if answer_budget <= 0:
                gc.collect()
                return ""

            p2 = og.GeneratorParams(handle.model)
            p2.set_search_options(
                max_length=min(len(phase2_tokens) + answer_budget, effective_ctx_limit),
                temperature=temp,
            )
            g2 = og.Generator(handle.model, p2)
            try:
                g2.append_tokens(phase2_tokens)
                answer_tokens: list = []
                printed_chars = 0
                while not g2.is_done() and len(answer_tokens) < answer_budget:
                    g2.generate_next_token()
                    answer_tokens.append(g2.get_next_tokens()[0])
                    # Stream decoded delta every 4 tokens to balance latency vs decode cost
                    if on_token and len(answer_tokens) % 4 == 0:
                        full = handle.tokenizer.decode(answer_tokens)
                        delta = full[printed_chars:]
                        if delta:
                            on_token(delta)
                            printed_chars = len(full)
                response = handle.tokenizer.decode(answer_tokens)
                # Flush any remaining decoded characters not yet streamed
                if on_token:
                    clean = strip_think(response).strip()
                    delta = clean[printed_chars:]
                    if delta:
                        on_token(delta)
                return strip_think(response).strip()
            finally:
                del g2
                del p2
                gc.collect()

        else:
            # ── Non-thinking model: single phase ─────────────────────────
            available_gen = min(max_tokens, effective_ctx_limit - len(input_tokens) - 10)
            if available_gen <= 0:
                gc.collect()
                return ""

            params = og.GeneratorParams(handle.model)
            params.set_search_options(
                max_length=min(len(input_tokens) + available_gen, effective_ctx_limit),
                temperature=temp,
            )
            generator = og.Generator(handle.model, params)
            try:
                generator.append_tokens(input_tokens)
                output_tokens: list = []
                printed_chars = 0
                while not generator.is_done() and len(output_tokens) < available_gen:
                    generator.generate_next_token()
                    output_tokens.append(generator.get_next_tokens()[0])
                    if on_token and len(output_tokens) % 4 == 0:
                        full = handle.tokenizer.decode(output_tokens)
                        delta = full[printed_chars:]
                        if delta:
                            on_token(delta)
                            printed_chars = len(full)
                response = handle.tokenizer.decode(output_tokens)
                if on_token:
                    delta = response.strip()[printed_chars:]
                    if delta:
                        on_token(delta)
                return response.strip()
            finally:
                del generator
                del params
                gc.collect()

    def generate_supervisor(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        think_budget: int = THINK_BUDGET_T2,
        disable_thinking: bool = False,
        on_token: Optional[callable] = None,
    ) -> str:
        if not self._supervisor_handle:
            raise RuntimeError("Supervisor not loaded")
        debug_log("gen.supervisor.begin", max_tokens=max_tokens)
        t0 = time.time()
        result = self._generate_with(
            self._supervisor_handle,
            prompt,
            system,
            max_tokens,
            temperature,
            think_budget,
            disable_thinking,
            on_token=on_token,
        )
        debug_log("gen.supervisor.end", elapsed_s=time.time()-t0)
        return result

    def generate_specialist(
            self, prompt: str, system: str = "",
            max_tokens: int = OUTPUT_BUDGET_T2,
            temperature: float = 0.9,
            on_token: Optional[callable] = None,
            loop_guard: bool = True) -> str:
        """Draft generation call — creative sampling at T=0.9 by default."""
        if not self._specialist_path or not self._specialist_path.startswith("ollama://"):
            raise RuntimeError("Specialist target is not an Ollama endpoint.")
        debug_log("gen.specialist.begin", max_tokens=max_tokens)
        result, _ = self._generate_via_ollama(
            self._specialist_path, prompt, system, max_tokens, temperature, on_token=on_token,
            loop_guard=loop_guard
        )
        # Strip Ollama `<think>` blocks if they exist natively.
        return strip_think(result)

    def generate_specialist_grade(
            self, prompt: str, system: str = "",
            max_tokens: int = 128) -> str:
        """Grader call — near-deterministic at T=0.01, short budget, fresh context."""
        if not self._specialist_path or not self._specialist_path.startswith("ollama://"):
            raise RuntimeError("Specialist target is not an Ollama endpoint.")
        debug_log("gen.specialist.grade.begin", max_tokens=max_tokens)
        result, _ = self._generate_via_ollama(
            self._specialist_path, prompt, system, max_tokens, temperature=0.01
        )
        return strip_think(result)

    def supervisor_ctx_limit(self) -> int:
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
# ═══════════════════════════════════════════════════════════════════════════




