#!/usr/bin/env python3
"""
bench_mmlu_pro_calc.py
======================
MMLU-Pro benchmark runner with <calc>...</calc> streaming injection.

The model runs in think_off mode (closed <think></think> prefill).
When the model emits a <calc>expression</calc> block mid-generation the
stream is interrupted, the expression is evaluated by a safe stdlib-math
evaluator, and "[CALC: result]" is injected into the assistant prefill
before generation resumes.  This loop repeats up to MAX_CALC_CALLS times
per question.

Implements the "cognitive division of labour" from COE_SPEC.md
§ Scientific Calculator Tool (May 9 2026):
  - Specialist model  : selects formula, substitutes values, interprets result
  - <calc> evaluator  : executes arithmetic deterministically
  - No coding expert needed — all expressions are scalar stdlib-math

Usage
-----
# Physics merged specialist on off5 OOD sample:
python bench_mmlu_pro_calc.py \\
    --category physics \\
    --model physics_merged \\
    --source data/ood_samples/mmlu_pro_ood_physics_off5.jsonl \\
    --out    data/matrix/bench_calc_physics_merged_off5.jsonl

# Parent Q4 on the same set (baseline arm of the 2x2):
python bench_mmlu_pro_calc.py \\
    --category physics \\
    --model parent_q4 \\
    --source data/ood_samples/mmlu_pro_ood_physics_off5.jsonl \\
    --out    data/matrix/bench_calc_parent_q4_physics_off5.jsonl

# Smoke test (first 5 questions):
python bench_mmlu_pro_calc.py \\
    --category physics --model parent_q4 \\
    --source data/ood_samples/mmlu_pro_ood_physics_off5.jsonl \\
    --out /tmp/smoke_calc.jsonl --smoke

Output schema (superset of bench_mmlu_pro.py)
---------------------------------------------
{
  "question_id":      int,
  "category":         str,
  "src":              str,
  "model":            str,
  "mode":             "calc_off",      # fixed: think_off + calc injection
  "attempt":          int,
  "temperature":      float,
  "repeat_penalty":   float,
  "num_ctx":          int,
  "question":         str,
  "options":          list[str],
  "reference_answer": str,             # "A".."J"
  "answer_index":     int,             # 0-based
  "loop_detected":    bool,
  "think_body":       "",              # always empty (think_off)
  "final_answer":     str,             # full assembled text with injections
  "candidate_letter": str | null,
  "correct":          bool | null,
  "elapsed_s":        float,
  "calc_calls":       int,             # number of <calc> blocks resolved
  "calc_log":         list[dict],      # [{round, expr, result_str, ok: bool}]
}
"""

import argparse
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, "data")
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Gemma4 raw chat-template tokens for /api/generate raw:true
_GEMMA4_BOS        = "<bos>"
_GEMMA4_USER_START = "<start_of_turn>user\n"
_GEMMA4_MODEL_START = "<start_of_turn>model\n"
_GEMMA4_TURN_END   = "<end_of_turn>\n"

MODEL_TAGS = {
    # MMLU off4+off9 merged specialists (primary subjects for calc experiment)
    "mmlu_math_merged":        "gemma4-math-mmlu-merged-k64:latest",
    "mmlu_physics_merged":     "gemma4-physics-mmlu-merged-k64:latest",
    "mmlu_engineering_merged": "gemma4-engineering-mmlu-merged-k64:latest",
    "mmlu_chemistry_merged":   "gemma4-chemistry-mmlu-merged-k64:latest",
    "mmlu_biology_merged":     "gemma4-biology-mmlu-merged-k64:latest",
    "mmlu_health_merged":      "gemma4-health-mmlu-merged-k64:latest",
    "mmlu_economics_merged":   "gemma4-economics-mmlu-merged-k64:latest",
    "mmlu_business_merged":    "gemma4-business-mmlu-merged-k64:latest",
    "mmlu_psychology_merged":  "gemma4-psychology-mmlu-merged-k64:latest",
    "mmlu_law_merged":         "gemma4-law-mmlu-merged-k64:latest",
    "mmlu_history_merged":     "gemma4-history-mmlu-merged-k64:latest",
    "mmlu_philosophy_merged":  "gemma4-philosophy-mmlu-merged-k64:latest",
    "mmlu_other_merged":       "gemma4-other-mmlu-merged-k64:latest",
    # Convenience aliases for common experiment arms
    "physics_merged":     "gemma4-physics-mmlu-merged-k64:latest",
    "math_merged":        "gemma4-math-mmlu-merged-k64:latest",
    "chemistry_merged":   "gemma4-chemistry-mmlu-merged-k64:latest",
    "engineering_merged": "gemma4-engineering-mmlu-merged-k64:latest",
    # Older single-corpus specialists
    "physics":     "gemma4-physics-b-supb-k64:latest",
    "math":        "gemma4-math-k64:latest",
    "engineering": "gemma4-engineering-a-v2-supplemented-k64:latest",
    # Parent baseline
    "parent_q4": "gemma4:26b-a4b-it-q4_K_M",
    "parent_q8": "gemma4:26b-a4b-it-q8_0",
}

MODEL_OPTIONS = {
    "temperature":    0.6,
    "top_k":          64,
    "top_p":          0.95,
    "repeat_penalty": 1.05,
    "repeat_last_n":  1024,
}

NUM_CTX     = 8192
NUM_PREDICT = 16000
GEN_TIMEOUT = 600       # 10 min per question
MAX_CALC_CALLS = 8      # guard against infinite <calc> loops

THINK_OFF_PREFIX = "<think></think>\n"

# Domains that receive the <calc> system prompt addendum.
# Other domains can still use --force-calc to override.
CALC_DOMAINS = {"physics", "math", "chemistry", "engineering"}

CALC_ADDENDUM = (
    "\n\nYou have access to an exact arithmetic evaluator."
    " Strategy — symbolic first, ONE calculation at the end:\n\n"
    "1. Work ENTIRELY symbolically throughout. Derive every relationship "
    "algebraically, keeping all quantities as symbols. Do NOT evaluate any "
    "numbers yourself at any step.\n"
    "2. Only when you have a single fully-substituted expression — with ALL "
    "constants, variables, and physical constants replaced by their numerical "
    "values — emit it once:\n\n"
    "       <calc>FULLY_SUBSTITUTED_EXPRESSION</calc>\n\n"
    "   The exact result will be injected inline. Identify which answer option "
    "it corresponds to and state your final answer.\n"
    "3. Compose every intermediate sub-expression into one single <calc> call "
    "rather than evaluating steps separately.\n"
    "4. Use at most ONE <calc> block.\n"
    "5. Allowed inside <calc>: numbers, +, -, *, /, **, (), sin, cos, tan, "
    "asin, acos, atan, sqrt, log, log10, exp, abs, pi, e. Use ** for "
    "exponents, not ^.\n"
    "6. If no numerical computation is needed, answer directly without <calc>.\n"
    "7. Never estimate or compute arithmetic yourself."
)

CATEGORY_PROMPTS = {
    "math": (
        "You are an expert mathematician. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "physics": (
        "You are an expert physicist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "engineering": (
        "You are an expert engineer. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "cs": (
        "You are an expert in computer science. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "chemistry": (
        "You are an expert chemist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "biology": (
        "You are an expert biologist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "health": (
        "You are an expert in health sciences and medicine. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "economics": (
        "You are an expert economist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "business": (
        "You are an expert in business and management. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "psychology": (
        "You are an expert psychologist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "law": (
        "You are an expert in law. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "history": (
        "You are an expert historian. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "philosophy": (
        "You are an expert philosopher. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
    "other": (
        "You are a knowledgeable expert. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        r"in \boxed{} — for example \boxed{C}. Write only the letter inside \boxed{}."
    ),
}

LETTERS = list("ABCDEFGHIJ")

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator
# ---------------------------------------------------------------------------

_EVAL_SAFE: dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_EVAL_SAFE["abs"] = abs

# Characters permitted in a <calc> expression (defence-in-depth before eval).
# Allows: digits, operators +-*/, parens, decimal point, comma (for atan2),
#         space, letters and underscore (function/constant names).
_EXPR_ALLOWED = re.compile(r"^[0-9+\-*/()., a-zA-Z_]+$")

# Detect a complete <calc>...</calc> block (non-greedy, DOTALL).
_CALC_FULL_RE = re.compile(r"<calc>(.*?)</calc>", re.DOTALL | re.IGNORECASE)


def safe_calc(expr: str) -> tuple[str, bool]:
    """Evaluate a <calc> expression.

    Returns (result_str, ok).  On rejection or error returns
    (error_description, False) so the caller can inject a meaningful
    error token and continue rather than crashing.
    """
    # Normalise: caret ^ → ** (models often write 2^3)
    expr = expr.replace("^", "**")
    expr = expr.strip()

    if not _EXPR_ALLOWED.match(expr):
        return f"REJECTED:{expr!r}", False

    try:
        result = eval(expr, {"__builtins__": {}}, _EVAL_SAFE)  # noqa: S307
    except Exception as exc:
        return f"ERROR:{exc}", False

    # Guard against non-scalar results
    if not isinstance(result, (int, float)):
        return f"TYPE:{type(result).__name__}", False

    # Represent inf/nan explicitly so the model knows something went wrong
    if math.isnan(result):
        return "NaN", True
    if math.isinf(result):
        return "Inf" if result > 0 else "-Inf", True

    # Prefer integer display for exact integers, otherwise use g format
    if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
        return str(int(result)), True

    return f"{result:.6g}", True


# ---------------------------------------------------------------------------
# Utility functions (self-contained — no import from bench_mmlu_pro)
# ---------------------------------------------------------------------------

LOOP_GUARD_MATCH  = 100
LOOP_GUARD_WINDOW = 400
LOOP_GUARD_MIN    = 1400


def _check_loop(text: str) -> bool:
    if len(text) < LOOP_GUARD_MIN:
        return False
    tail   = text[-LOOP_GUARD_MATCH:]
    window = text[-(LOOP_GUARD_MATCH + LOOP_GUARD_WINDOW):-LOOP_GUARD_MATCH]
    return tail in window


_BOXED_RE      = re.compile(r"\\boxed\s*\{")
_HTML_BOXED_RE = re.compile(r"<boxed>\s*([A-Ja-j])\s*</boxed>", re.IGNORECASE)
_HTML_BOX_RE   = re.compile(r"<box>\s*([A-Ja-j])\s*</box>",    re.IGNORECASE)


def _extract_candidate_letter(text: str) -> str | None:
    if not text:
        return None

    # 1. LaTeX \boxed{X}
    positions = [m.start() for m in _BOXED_RE.finditer(text)]
    if positions:
        pos = positions[-1]
        brace_start = text.index("{", pos)
        depth, inner = 0, None
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    inner = text[brace_start + 1 : i].strip()
                    break
        if inner and inner.upper() in LETTERS:
            return inner.upper()

    # 2. HTML <boxed>X</boxed>
    hits = list(_HTML_BOXED_RE.finditer(text))
    if hits:
        letter = hits[-1].group(1).upper()
        if letter in LETTERS:
            return letter

    # 3. HTML <box>X</box>
    hits = list(_HTML_BOX_RE.finditer(text))
    if hits:
        letter = hits[-1].group(1).upper()
        if letter in LETTERS:
            return letter

    # 4. Freeform fallback patterns
    patterns = [
        r"\bthe\s+(?:answer|correct\s+option)\s+is\s+([A-Ja-j])\b",
        r"\bfinal\s+answer\s*[:\-]\s*\(?([A-Ja-j])\)?",
        r"\banswer\s*[:\-]\s*\(?([A-Ja-j])\)?",
        r"\(([A-Ja-j])\)\s*$",
        r"^([A-Ja-j])\s*$",
    ]
    for pat in patterns:
        for m in reversed(list(re.finditer(pat, text, re.IGNORECASE | re.MULTILINE))):
            letter = m.group(1).upper()
            if letter in LETTERS:
                return letter

    # 5. Bold **X** at end of text
    m = re.search(r"\*{1,2}([A-Ja-j])\*{1,2}\s*$", text, re.IGNORECASE)
    if m and m.group(1).upper() in LETTERS:
        return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Numerical closest-match (used when a <calc> result is available)
# Does NOT modify _extract_candidate_letter — that function is unchanged.
# ---------------------------------------------------------------------------

def _parse_option_number(text: str) -> float | None:
    """Extract the primary numerical value from an MC option string.

    Handles:
      "9.4 km"                          → 9.4
      "3.8 × 10^{-3} statV/cm"          → 3.8e-3
      "$3.2 \\times 10^{-3}$"            → 3.2e-3
      "2.5 million light years"          → 2.5   (units stripped)
      "1.2e5"                            → 1.2e5
    """
    t = re.sub(r'[$\\]', ' ', text).strip()
    # × 10^N  or  × 10^{N}  (LaTeX or plain unicode ×)
    m = re.search(
        r'(-?\d+\.?\d*)\s*[×x]\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?',
        t, re.IGNORECASE,
    )
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    # plain float / scientific notation
    m = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', t)
    if m:
        return float(m.group(0))
    return None


def _match_calc_to_option(calc_result: float, options: list[str]) -> str | None:
    """Return the MC letter (A–J) whose option value is numerically closest to
    calc_result.  Returns None when fewer than half the options parse as numbers
    (signals caller to fall back to text-based extraction)."""
    parsed: list[tuple[int, float]] = []
    for i, opt in enumerate(options):
        val = _parse_option_number(opt)
        if val is not None:
            parsed.append((i, val))

    if len(parsed) < len(options) // 2:
        return None   # mostly non-numerical options; text extractor is better

    def _rel_err(v: float) -> float:
        denom = max(abs(calc_result), abs(v), 1e-30)
        return abs(v - calc_result) / denom

    best_i, best_err = min(parsed, key=lambda x: (_rel_err(x[1]), x[0]))
    # Reject the numerical match when it's nowhere close — the last <calc> may
    # compute an intermediate value (e.g. field strength in SI) while the MC
    # options list a converted or paired value.  Fall back to text extractor.
    if best_err > 0.30:
        return None
    return LETTERS[best_i] if best_i < len(LETTERS) else None


def format_question(q: dict) -> str:
    lines = [q["question"], ""]
    for i, opt in enumerate(q["options"]):
        label = LETTERS[i] if i < len(LETTERS) else str(i)
        lines.append(f"{label}. {opt}")
    return "\n".join(lines)


def load_done(out_path: str, attempt: int) -> set[tuple[int, str, str, int]]:
    done: set[tuple[int, str, str, int]] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["question_id"], r["model"], r["mode"], r.get("attempt", 1)))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_result(out_path: str, record: dict) -> None:
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core: streaming call with <calc> injection loop
# ---------------------------------------------------------------------------

def _build_raw_prompt(system_prompt: str, question: str) -> str:
    """Return the full Gemma4 raw prompt up to (and including) the think-off prefix.

    NOTE: <bos> is required here.  Gemma4 GGUF has add_bos_token=false, so
    Ollama's llama.cpp backend does NOT add BOS automatically in raw mode.
    Omitting it leaves the model without a beginning-of-sequence token and
    causes instability on harder questions.
    """
    return (
        _GEMMA4_BOS
        + _GEMMA4_USER_START
        + system_prompt + "\n\n"
        + question
        + _GEMMA4_TURN_END
        + _GEMMA4_MODEL_START
        + THINK_OFF_PREFIX          # "<think></think>\n"
    )


def ollama_stream_calc(
    model_tag: str,
    prompt: str,
    system_prompt: str,
    progress_prefix: str = "",
) -> tuple[str, bool, float, int, list[dict]]:
    """Stream a model response with true inline <calc> injection.

    Uses /api/generate raw:true so the calc result is injected directly into
    the ongoing generation context — the model never closes its assistant turn
    and never sees a new user message.  Each round extends raw_prompt with:
        accumulated_text_up_to_</calc> + " = RESULT. "
    and the model continues from there within the same <start_of_turn>model block.

    Returns:
        assembled   : all text the model produced (includes inline injections)
        loop_detected : bool
        elapsed_s   : wall-clock seconds
        calc_calls  : number of <calc> blocks evaluated
        calc_log    : list of {round, expr, result_str, ok}
    """
    # Fixed context prefix: user turn + think-off.  raw_prompt grows each round
    # by appending what the model generated + the injected result.
    raw_prompt     = _build_raw_prompt(system_prompt, prompt)
    assembled_text = ""   # everything generated (including injections, excl. base_prompt)

    calc_log: list[dict] = []
    loop_detected = False
    t0 = time.time()

    for round_idx in range(MAX_CALC_CALLS + 1):
        final_round = (round_idx == MAX_CALC_CALLS)
        if final_round:
            print(f"\n{progress_prefix}[CALC] max rounds reached, final pass", flush=True)

        body = {
            "model":   model_tag,
            "prompt":  raw_prompt,
            "stream":  True,
            "raw":     True,        # pass prompt verbatim — no template wrapping
            "think":   False,
            "options": {
                **MODEL_OPTIONS,
                "num_ctx":     NUM_CTX,
                "num_predict": NUM_PREDICT,
                "stop":        ["<end_of_turn>"],   # Gemma4 turn-end token
            },
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_GENERATE_URL, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        accumulated = ""   # tokens generated in this round only
        calc_found  = False

        try:
            with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as resp:
                print()
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # /api/generate uses "response", not "message.content"
                    token = chunk.get("response") or ""
                    if token:
                        accumulated += token
                        print(token, end="", flush=True)

                        # Loop guard across all generated text
                        if _check_loop(assembled_text + accumulated):
                            loop_detected = True
                            print(f"\n{progress_prefix}[LOOP-ABORT at "
                                  f"{len(assembled_text)+len(accumulated)} chars]",
                                  flush=True)
                            break

                        # Detect a complete <calc>...</calc> block — stop the stream
                        if not final_round and "</calc>" in accumulated.lower():
                            m = _CALC_FULL_RE.search(accumulated)
                            if m:
                                calc_found = True
                                break

                    if chunk.get("done"):
                        break

        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unreachable: {exc}") from exc

        if loop_detected:
            return assembled_text + accumulated, True, round(time.time() - t0, 1), \
                   len(calc_log), calc_log

        if not calc_found:
            return assembled_text + accumulated, False, round(time.time() - t0, 1), \
                   len(calc_log), calc_log

        # --- Process the first <calc>...</calc> found in this round ----------
        m    = _CALC_FULL_RE.search(accumulated)
        expr = m.group(1).strip()

        result_str, ok = safe_calc(expr)

        print(f"\n{progress_prefix}[CALC/{round_idx}] {expr!r} → {result_str}", flush=True)

        calc_log.append({
            "round":      round_idx,
            "expr":       expr,
            "result_str": result_str,
            "ok":         ok,
        })

        # Inline injection: extend raw_prompt with everything generated this
        # round up to and including </calc>, then the evaluated result.
        # The model continues within the same <start_of_turn>model block —
        # no turn boundary, no user message, no restart.
        text_to_end_of_calc = accumulated[:m.end()]
        inline_result       = f" = {result_str}. "

        raw_prompt     += text_to_end_of_calc + inline_result
        assembled_text += text_to_end_of_calc + inline_result
        # Loop continues: next round streams from the extended raw_prompt.

    elapsed = round(time.time() - t0, 1)
    return assembled_text, False, elapsed, len(calc_log), calc_log




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMLU-Pro benchmark with <calc> streaming injection."
    )
    parser.add_argument("--category", required=True,
                        choices=list(CATEGORY_PROMPTS.keys()),
                        help="MMLU-Pro category (selects system prompt).")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_TAGS.keys()),
                        help="Model key.")
    parser.add_argument("--source", required=True,
                        help="Input JSONL file (OOD sample or full split).")
    parser.add_argument("--out", required=True,
                        help="Output JSONL file (appended to, safe to resume).")
    parser.add_argument("--smoke", action="store_true",
                        help="First 5 questions only.")
    parser.add_argument("--qids", nargs="+", type=int, default=[],
                        help="Run specific question_id values only.")
    parser.add_argument("--attempt", type=int, default=1,
                        help="Pass@k round number (default 1).")
    parser.add_argument("--force-calc", action="store_true",
                        help="Add <calc> system prompt even for non-STEM domains.")
    parser.add_argument("--no-calc", action="store_true",
                        help="Disable <calc> injection (runs as plain think_off for comparison).")
    args = parser.parse_args()

    # Build system prompt
    base_prompt = CATEGORY_PROMPTS[args.category]
    use_calc = (not args.no_calc) and (args.category in CALC_DOMAINS or args.force_calc)
    system_prompt = base_prompt + (CALC_ADDENDUM if use_calc else "")

    model_tag = MODEL_TAGS[args.model]
    mode_name = "calc_off" if use_calc else "think_off_plain"

    print(f"[INFO] Category    : {args.category}")
    print(f"[INFO] Model       : {args.model}  ({model_tag})")
    print(f"[INFO] Mode        : {mode_name}")
    print(f"[INFO] Calc inject : {'ENABLED' if use_calc else 'DISABLED'}")
    print(f"[INFO] Source      : {args.source}")
    print(f"[INFO] Output      : {args.out}")

    if not os.path.exists(args.source):
        sys.exit(f"[ERROR] --source not found: {args.source}")

    with open(args.source, encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]
    print(f"[INFO] Questions   : {len(questions)}")

    if args.qids:
        qid_set   = set(args.qids)
        questions = [q for q in questions if q["question_id"] in qid_set]
        print(f"[INFO] QID filter  : {len(questions)} questions")

    if args.smoke:
        questions = questions[:5]
        print(f"[INFO] SMOKE TEST  : {len(questions)} questions")

    if not questions:
        sys.exit("[ERROR] No questions to run.")

    done = load_done(args.out, args.attempt)
    print(f"[INFO] Already done: {len(done)} records\n")

    n_done = n_correct = n_loop = n_errors = n_calc_used = 0

    for q in questions:
        qid = q["question_id"]
        key = (qid, args.model, mode_name, args.attempt)
        if key in done:
            print(f"  [{qid}] SKIP (done)")
            continue

        prompt   = format_question(q)
        ref_ans  = q["answer"]
        ref_idx  = q["answer_index"]
        preview  = q["question"][:60].replace("\n", " ")
        pfx      = f"  [qid={qid}] {args.model}/calc: "
        print(f"{pfx}{preview}...")

        try:
            assembled, loop_detected, elapsed, calc_calls, calc_log = ollama_stream_calc(
                model_tag=model_tag,
                prompt=prompt,
                system_prompt=system_prompt,
                progress_prefix=f"  [qid={qid}] ",
            )
        except RuntimeError as exc:
            print(f"    ERROR: {exc}", flush=True)
            n_errors += 1
            continue

        # The assembled text begins with THINK_OFF_PREFIX then the answer.
        # Strip the prefix to get the pure answer text for letter extraction.
        final_answer = assembled
        if final_answer.startswith(THINK_OFF_PREFIX):
            final_answer = final_answer[len(THINK_OFF_PREFIX):]

        # --- Answer extraction -------------------------------------------------
        # Primary (when calc used): numerical closest-match against MC options.
        # Fallback (always): existing text-based letter extractor (unchanged).
        calc_match_letter: str | None = None
        if not loop_detected and calc_calls > 0 and calc_log:
            try:
                last_val = float(calc_log[-1]["result_str"])
                calc_match_letter = _match_calc_to_option(last_val, q["options"])
            except (ValueError, KeyError, IndexError):
                pass

        # Try text extractor even on loops — the model often states the correct
        # answer clearly before the degenerate repetition begins.
        text_letter = _extract_candidate_letter(final_answer)

        # Loop-recovery supplement: _extract_candidate_letter requires ":" or "-"
        # after "answer", but a looping model often writes "Final Answer is D."
        # Try an additional "is [letter]" pattern without modifying the extractor.
        if text_letter is None:
            _m = re.search(
                r'\b(?:final\s+answer|answer)\s+is\s+([A-Ja-j])\b',
                final_answer, re.IGNORECASE,
            )
            if _m:
                text_letter = _m.group(1).upper()

        candidate_letter = calc_match_letter if calc_match_letter is not None else text_letter

        if candidate_letter is not None:
            correct = (candidate_letter == ref_ans)
        else:
            correct = None

        if correct is True:
            n_correct += 1
        if loop_detected:
            n_loop += 1
        if calc_calls > 0:
            n_calc_used += 1

        rec = {
            "question_id":      qid,
            "category":         q.get("category", args.category),
            "src":              q.get("src", ""),
            "model":            args.model,
            "mode":             mode_name,
            "attempt":          args.attempt,
            "temperature":      MODEL_OPTIONS["temperature"],
            "repeat_penalty":   MODEL_OPTIONS["repeat_penalty"],
            "num_ctx":          NUM_CTX,
            "question":         q["question"],
            "options":          q["options"],
            "reference_answer": ref_ans,
            "answer_index":     ref_idx,
            "loop_detected":    loop_detected,
            "think_body":       "",           # always empty (think_off)
            "final_answer":     final_answer,
            "calc_match_letter": calc_match_letter,
            "text_letter":      text_letter,
            "candidate_letter": candidate_letter,
            "correct":          correct,
            "elapsed_s":        elapsed,
            "calc_calls":       calc_calls,
            "calc_log":         calc_log,
        }
        append_result(args.out, rec)
        done.add(key)

        status = ("LOOP " if loop_detected else
                  "✓    " if correct is True  else
                  "✗    " if correct is False  else
                  "?box ")

        print(f"    {status}  got={candidate_letter}  ref={ref_ans}  "
              f"calc={calc_calls}  elapsed={elapsed}s", flush=True)
        n_done += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  MMLU-PRO {args.category.upper()} — {args.model} / {mode_name}")
    print(f"{'='*60}")
    print(f"  Generated : {n_done}")
    print(f"  Correct   : {n_correct}  ({100.0*n_correct/n_done:.1f}%)" if n_done else "  Correct   : -")
    print(f"  Loops     : {n_loop}")
    print(f"  Errors    : {n_errors}")
    print(f"  Used calc : {n_calc_used} / {n_done} questions")
    print(f"[OUT] {args.out}")


if __name__ == "__main__":
    main()
