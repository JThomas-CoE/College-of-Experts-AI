#!/usr/bin/env python3
"""
bench_mmlu_pro.py
=================
MMLU-Pro benchmark runner for Gemma4 specialist models.

Source: TIGER-Lab/MMLU-Pro — 10-choice MCQ across multiple categories.
Scoring: extract the letter answer (A-J) from model output, compare to reference.

Unlike AIME (integer answers), MMLU-Pro is MCQ, so the model is prompted to
state its final answer as a single letter inside \\boxed{} — e.g. \\boxed{C}.
Chain-of-thought is encouraged; the letter is extracted from the last \\boxed{}.

MMLU-Pro standard evaluation uses 5-shot CoT prompting.
We use 0-shot with constrained implicit CoT (think_off) which matches our AIME
methodology and is increasingly standard for reasoning-capable models.

Usage
-----
# Math subset, Q4 specialist, think_off (default category=math):
python bench_mmlu_pro.py --models math --modes think_off

# Physics specialist:
python bench_mmlu_pro.py --category physics --models physics --modes think_off

# Engineering + base comparison:
python bench_mmlu_pro.py --category engineering --models engineering parent_q4 --modes think_off

# Smoke test:
python bench_mmlu_pro.py --category physics --models physics --modes think_off --smoke

# Specific question ids:
python bench_mmlu_pro.py --category math --models math --modes think_off --qids 100 200 300

Output
------
data/bench_results_mmlu_pro_<category>.jsonl

RECORD SCHEMA
-------------
{
  "question_id":      int,
  "category":         str,
  "src":              str,
  "model":            str,
  "mode":             str,
  "attempt":          int,
  "temperature":      float,
  "repeat_penalty":   float,
  "num_ctx":          int,
  "question":         str,
  "options":          list[str],
  "reference_answer": str,   # letter "A".."J"
  "answer_index":     int,   # 0-based
  "loop_detected":    bool,
  "think_body":       str,
  "final_answer":     str | null,
  "candidate_letter": str | null,  # extracted letter from \\boxed{}
  "correct":          bool | null,
  "elapsed_s":        float,
}
"""

import argparse
import io
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_TAGS = {
    # Math
    "math":              "gemma4-math-k64:latest",
    "math_q8":           "gemma4-math-k64-q8:latest",
    # Physics
    "physics":           "gemma4-physics-b-supb-k64:latest",
    "physics_supa":      "gemma4-physics-b-supa-k64:latest",
    # Engineering
    "engineering":               "gemma4-engineering-a-v2-supplemented-k64:latest",
    "engineering_b_v2":          "gemma4-engineering-b-v2-k64:latest",
    # Med-pharma
    "medpharma":         "gemma4-medpharma-kbswap-b8-K64:latest",
    # Coding
    "python":            "gemma4-python-K64-q4_K_M:latest",
    "python_q8":         "gemma4-python-K64-q8_0:latest",
    "coding":            "gemma4-coding-K64-q4_K_M:latest",
    "web":               "gemma4-web-closure-ext-cap3-k64:latest",
    # Base models (shared across categories for comparison)
    "math_base":         "gemma4:26b-a4b-it-q8_0",
    "parent_q4":         "gemma4:26b-a4b-it-q4_K_M",
    # Cross-family comparison (dense Qwen3.5 vs MoE Gemma4 specialists)
    "qwen35_9b_q8":      "qwen3.5:9b-q8_0",
    "qwen35_9b_q4":      "qwen3.5:9b",
    "qwen35_4b_q4":      "qwen3.5:4b",
    # MMLU off4 activation-derived specialists (14 domains, swapactive K64)
    "mmlu_math":         "gemma4-math-mmlu-k64:latest",
    "mmlu_physics":      "gemma4-physics-mmlu-k64:latest",
    "mmlu_engineering":  "gemma4-engineering-mmlu-k64:latest",
    "mmlu_cs":           "gemma4-cs-mmlu-k64:latest",
    "mmlu_chemistry":    "gemma4-chemistry-mmlu-k64:latest",
    "mmlu_biology":      "gemma4-biology-mmlu-k64:latest",
    "mmlu_health":       "gemma4-health-mmlu-k64:latest",
    "mmlu_economics":    "gemma4-economics-mmlu-k64:latest",
    "mmlu_business":     "gemma4-business-mmlu-k64:latest",
    "mmlu_psychology":   "gemma4-psychology-mmlu-k64:latest",
    "mmlu_law":          "gemma4-law-mmlu-k64:latest",
    "mmlu_history":      "gemma4-history-mmlu-k64:latest",
    "mmlu_philosophy":   "gemma4-philosophy-mmlu-k64:latest",
    "mmlu_other":        "gemma4-other-mmlu-k64:latest",
    # MMLU off4+off9 merged specialists (15 models, swapactive K64)
    "mmlu_math_merged":        "gemma4-math-mmlu-merged-k64:latest",
    "mmlu_physics_merged":     "gemma4-physics-mmlu-merged-k64:latest",
    "mmlu_engineering_merged": "gemma4-engineering-mmlu-merged-k64:latest",
    "mmlu_cs_merged":          "gemma4-cs-mmlu-merged-k64:latest",
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
    "mmlu_other_merged_pu":    "gemma4-other-mmlu-merged-pu-k64:latest",
    "mmlu_other_merged_sqrtn": "gemma4-other-mmlu-merged-sqrtn-k64:latest",
}

MODEL_OPTIONS = {
    "temperature":    0.6,
    "top_k":          64,
    "top_p":          0.95,
    "repeat_penalty": 1.05,
    "repeat_last_n":  1024,
}

NUM_CTX_THINKOFF = 8192
NUM_CTX_THINKON  = 32768
NUM_PREDICT      = 16000

GEN_TIMEOUT = 600   # 10 min — MMLU-Pro problems are shorter than AIME

THINK_OFF_PREFIX = "<think></think>\n"

# Per-category configuration: (data_file_stem, system_prompt, output_file_stem)
CATEGORY_CONFIG = {
    "math": (
        "mmlu_pro_math",
        "You are an expert mathematician. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_math",
    ),
    "physics": (
        "mmlu_pro_physics",
        "You are an expert physicist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_physics",
    ),
    "engineering": (
        "mmlu_pro_engineering",
        "You are an expert engineer. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_engineering",
    ),
    "cs": (
        "mmlu_pro_cs",
        "You are an expert in computer science. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_cs",
    ),
    "law": (
        "mmlu_pro_law",
        "You are an expert in law. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_law",
    ),
    "health": (
        "mmlu_pro_health",
        "You are an expert in health sciences and medicine. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_health",
    ),
    "economics": (
        "mmlu_pro_economics",
        "You are an expert economist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_economics",
    ),
    "business": (
        "mmlu_pro_business",
        "You are an expert in business and management. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_business",
    ),
    "psychology": (
        "mmlu_pro_psychology",
        "You are an expert psychologist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_psychology",
    ),
    "history": (
        "mmlu_pro_history",
        "You are an expert historian. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_history",
    ),
    "philosophy": (
        "mmlu_pro_philosophy",
        "You are an expert philosopher. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_philosophy",
    ),
    "biology": (
        "mmlu_pro_biology",
        "You are an expert biologist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_biology",
    ),
    "chemistry": (
        "mmlu_pro_chemistry",
        "You are an expert chemist. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_chemistry",
    ),
    "other": (
        "mmlu_pro_other",
        "You are a knowledgeable expert. You will be given a multiple-choice question "
        "with options labeled A through J. Work through the problem step by step, showing "
        "your reasoning. At the end, state your final answer as a single letter enclosed "
        "in \\boxed{} — for example \\boxed{C}. Write only the letter inside \\boxed{}.",
        "bench_results_mmlu_pro_other",
    ),
}

# Runtime-resolved (set in main after --category is parsed)
SYSTEM_PROMPT = CATEGORY_CONFIG["math"][1]

LETTERS = list("ABCDEFGHIJ")

# ---------------------------------------------------------------------------
# Loop guard
# ---------------------------------------------------------------------------

LOOP_GUARD_MATCH  = 100
LOOP_GUARD_WINDOW = 400
LOOP_GUARD_MIN    = 1400

def _check_loop(assembled: str) -> bool:
    if len(assembled) < LOOP_GUARD_MIN:
        return False
    tail   = assembled[-LOOP_GUARD_MATCH:]
    window = assembled[-(LOOP_GUARD_MATCH + LOOP_GUARD_WINDOW) : -LOOP_GUARD_MATCH]
    return tail in window


def _split_think(assembled: str) -> tuple[str, str]:
    m_open  = re.search(r"<think>",  assembled, re.IGNORECASE)
    m_close = re.search(r"</think>", assembled, re.IGNORECASE)
    if m_open and m_close and m_close.end() > m_open.start():
        think_body = assembled[m_open.start() : m_close.end()]
        answer     = assembled[m_close.end():].strip()
    elif m_open:
        think_body = assembled[m_open.start():]
        answer     = ""
    else:
        think_body = ""
        answer     = assembled.strip()
    return think_body, answer


# ---------------------------------------------------------------------------
# Letter extraction
# ---------------------------------------------------------------------------

_BOXED_RE     = re.compile(r'\\boxed\s*\{')
_HTML_BOXED_RE = re.compile(r'<boxed>\s*([A-Ja-j])\s*</boxed>', re.IGNORECASE)
_HTML_BOX_RE   = re.compile(r'<box>\s*([A-Ja-j])\s*</box>',   re.IGNORECASE)

def _extract_candidate_letter(text: str) -> str | None:
    """Extract the last answer letter from model output.

    Priority order:
      1. LaTeX \\boxed{X}  — prompted format
      2. HTML <boxed>X</boxed> — common model deviation
      3. HTML <box>X</box>    — abbreviated variant
      4. Freeform fallback patterns (answer is X, Final answer: X, etc.)
      5. Bold **X** at end of text
    """
    if not text:
        return None

    # 1. LaTeX \boxed{X}
    positions = [m.start() for m in _BOXED_RE.finditer(text)]
    if positions:
        pos = positions[-1]
        brace_start = text.index('{', pos)
        depth = 0
        inner = None
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    inner = text[brace_start + 1 : i].strip()
                    break
        if inner:
            letter = inner.upper().strip()
            if letter in LETTERS:
                return letter

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
        r'\bthe\s+(?:answer|correct\s+option)\s+is\s+([A-Ja-j])\b',
        r'\bfinal\s+answer\s*[:\-]\s*\(?([A-Ja-j])\)?',
        r'\banswer\s*[:\-]\s*\(?([A-Ja-j])\)?',
        r'\(([A-Ja-j])\)\s*$',
        r'^([A-Ja-j])\s*$',
    ]
    for pat in patterns:
        for m in reversed(list(re.finditer(pat, text, re.IGNORECASE | re.MULTILINE))):
            letter = m.group(1).upper()
            if letter in LETTERS:
                return letter

    # 5. Bold **X** or *X* isolated at end of text (last non-whitespace token)
    m = re.search(r'\*{1,2}([A-Ja-j])\*{1,2}\s*$', text, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if letter in LETTERS:
            return letter

    return None


# ---------------------------------------------------------------------------
# Question formatter
# ---------------------------------------------------------------------------

def format_question(q: dict) -> str:
    """Format question + labeled options as a prompt string."""
    options = q["options"]
    lines = [q["question"], ""]
    for i, opt in enumerate(options):
        label = LETTERS[i] if i < len(LETTERS) else str(i)
        lines.append(f"{label}. {opt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_done(out_path: str, attempt: int = 1) -> set[tuple[int, str, str, int]]:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rec_attempt = r.get("attempt", 1)
                done.add((r["question_id"], r["model"], r["mode"], rec_attempt))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_result(out_path: str, record: dict) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Ollama streaming call
# ---------------------------------------------------------------------------

def ollama_stream(
    model_tag: str,
    prompt: str,
    think_off: bool,
    progress_prefix: str = "",
) -> tuple[str, bool, float]:
    num_ctx = NUM_CTX_THINKOFF if think_off else NUM_CTX_THINKON

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    if think_off:
        messages.append({"role": "assistant", "content": THINK_OFF_PREFIX})

    body = {
        "model":    model_tag,
        "stream":   True,
        "think":    not think_off,
        "messages": messages,
        "options": {
            **MODEL_OPTIONS,
            "num_ctx":     num_ctx,
            "num_predict": NUM_PREDICT,
        },
    }

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    think_assembled  = ""
    answer_assembled = ""
    loop_detected    = False

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
                msg          = chunk.get("message", {})
                think_token  = msg.get("thinking", "") or ""
                answer_token = msg.get("content",  "") or ""
                if think_token:
                    think_assembled += think_token
                    print(think_token, end="", flush=True)
                    if _check_loop(think_assembled):
                        loop_detected = True
                        print(f"\n{progress_prefix}[LOOP-ABORT in think at {len(think_assembled)} chars]",
                              flush=True)
                        break
                if answer_token:
                    answer_assembled += answer_token
                    print(answer_token, end="", flush=True)
                    if _check_loop(answer_assembled):
                        loop_detected = True
                        print(f"\n{progress_prefix}[LOOP-ABORT in answer at {len(answer_assembled)} chars]",
                              flush=True)
                        break
                if chunk.get("done"):
                    break
            print()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unreachable: {exc}") from exc

    elapsed = round(time.time() - t0, 1)

    if think_assembled:
        assembled = f"<think>{think_assembled}</think>\n{answer_assembled}"
    else:
        assembled = answer_assembled

    return assembled, loop_detected, elapsed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(category: str, val: bool = False) -> list[dict]:
    data_stem = CATEGORY_CONFIG[category][0]
    fname = f"{data_stem}_val.jsonl" if val else f"{data_stem}.jsonl"
    path  = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Questions file not found: {path}\n"
                 f"        Run _download_mmlu_pro.py --category {category} first.")
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(records: list[dict], category: str = "math") -> None:
    models = sorted({r["model"] for r in records})
    modes  = sorted({r["mode"]  for r in records})
    print(f"\n{'='*60}")
    print(f"  MMLU-PRO {category.upper()} RESULTS SUMMARY")
    print(f"{'='*60}")
    for model in models:
        for mode in modes:
            sub = [r for r in records if r["model"] == model and r["mode"] == mode]
            if not sub:
                continue
            correct   = [r for r in sub if r.get("correct") is True]
            wrong     = [r for r in sub if r.get("correct") is False]
            no_answer = [r for r in sub if r.get("correct") is None]
            loops     = [r for r in sub if r.get("loop_detected")]
            n         = len(sub)
            print(f"\n  {model} / {mode}")
            print(f"    Overall   : {len(correct)}/{n}  ({100.0*len(correct)/n:.1f}%)")
            print(f"    Loops     : {len(loops)}")
            print(f"    No answer : {len(no_answer)}")
            if wrong:
                print(f"    Wrong     : {len(wrong)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MMLU-Pro benchmark runner.")
    parser.add_argument("--category", choices=list(CATEGORY_CONFIG.keys()),
                        default="math",
                        help="MMLU-Pro category to benchmark (default: math).")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_TAGS.keys()),
                        default=None,
                        help="Model key(s) to run. Default per category.")
    parser.add_argument("--modes", nargs="+", choices=["think_on", "think_off"],
                        default=["think_off"],
                        help="Think mode(s).")
    parser.add_argument("--smoke", action="store_true",
                        help="First 20 questions only.")
    parser.add_argument("--val", action="store_true",
                        help="Use validation split instead of test split.")
    parser.add_argument("--qids", nargs="+", type=int, default=[],
                        help="Run only specific question_id values.")
    parser.add_argument("--out", default="",
                        help="Override output JSONL path.")
    parser.add_argument("--attempt", type=int, default=1,
                        help="Pass@k round number (1-based).")
    parser.add_argument("--source", default="",
                        help="Override input JSONL path (e.g. an OOD sample file). "
                             "Category key is still required for system prompt selection.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None, dest="repeat_penalty")
    parser.add_argument("--num-ctx-thinkon", type=int, default=None, dest="num_ctx_thinkon")
    parser.add_argument("--num-predict", type=int, default=None, dest="num_predict")
    args = parser.parse_args()

    global NUM_CTX_THINKON, NUM_PREDICT, MODEL_OPTIONS, SYSTEM_PROMPT
    if args.num_ctx_thinkon is not None:
        NUM_CTX_THINKON = args.num_ctx_thinkon
    if args.num_predict is not None:
        NUM_PREDICT = args.num_predict
    if args.temperature is not None:
        MODEL_OPTIONS = {**MODEL_OPTIONS, "temperature": args.temperature}
    if args.repeat_penalty is not None:
        MODEL_OPTIONS = {**MODEL_OPTIONS, "repeat_penalty": args.repeat_penalty}

    category = args.category
    _, cat_prompt, out_stem = CATEGORY_CONFIG[category]
    SYSTEM_PROMPT = cat_prompt

    # Default models per category
    category_default_models = {
        "math":        ["math"],
        "physics":     ["physics"],
        "engineering": ["engineering"],
        "cs":          ["parent_q4"],
        "law":         ["parent_q4"],
        "health":      ["medpharma"],
        "economics":   ["parent_q4"],
        "business":    ["parent_q4"],
        "psychology":  ["parent_q4"],
        "history":     ["parent_q4"],
        "philosophy":  ["parent_q4"],
        "biology":     ["parent_q4"],
        "chemistry":   ["parent_q4"],
        "other":       ["parent_q4"],
    }
    model_keys = args.models or category_default_models.get(category, ["parent_q4"])

    out_path = args.out or os.path.join(DATA_DIR, f"{out_stem}.jsonl")
    attempt  = args.attempt

    print(f"[INFO] Category: {category}")
    print(f"[INFO] Models  : {model_keys}")
    print(f"[INFO] Modes   : {args.modes}")
    print(f"[INFO] Attempt : {attempt}")
    print(f"[INFO] Output  : {out_path}")

    if args.source:
        if not os.path.exists(args.source):
            sys.exit(f"[ERROR] --source file not found: {args.source}")
        with open(args.source, encoding="utf-8") as _sf:
            questions = [json.loads(l) for l in _sf if l.strip()]
        print(f"[INFO] Questions loaded from --source: {len(questions)}  ({args.source})")
    else:
        questions = load_questions(category, val=args.val)
    print(f"[INFO] Questions loaded: {len(questions)}")

    if args.qids:
        qid_set   = set(args.qids)
        questions = [q for q in questions if q["question_id"] in qid_set]
        print(f"[INFO] QID filter — {len(questions)} questions.")

    if args.smoke:
        questions = questions[:20]
        print(f"[INFO] SMOKE TEST — {len(questions)} questions.")

    if not questions:
        sys.exit("[ERROR] No questions to run.")

    done = load_done(out_path, attempt)
    print(f"[INFO] Already done: {len(done)} records\n")

    n_done = n_correct = n_loop = n_errors = 0

    for mode in args.modes:
        think_off = (mode == "think_off")
        for model_key in model_keys:
            model_tag = MODEL_TAGS[model_key]
            total_q   = len(questions)
            print(f"\n{'='*60}")
            print(f"  MODEL: {model_key}  ({model_tag})")
            print(f"  MODE : {mode}  ({total_q} questions)")
            print(f"{'='*60}\n")

            for q in questions:
                qid    = q["question_id"]
                key    = (qid, model_key, mode, attempt)
                if key in done:
                    print(f"  [{qid}] SKIP (done)")
                    continue

                prompt  = format_question(q)
                ref_ans = q["answer"]          # letter "A".."J"
                ref_idx = q["answer_index"]    # 0-based
                preview = q["question"][:60].replace("\n", " ")
                progress = f"  [qid={qid}] {model_key}/{mode}: "
                print(f"{progress}{preview}...")

                try:
                    assembled, loop_detected, elapsed = ollama_stream(
                        model_tag=model_tag,
                        prompt=prompt,
                        think_off=think_off,
                        progress_prefix=f"  [qid={qid}] ",
                    )
                except RuntimeError as exc:
                    print(f"    ERROR: {exc}", flush=True)
                    n_errors += 1
                    continue

                think_body, final_answer = _split_think(assembled)

                if loop_detected:
                    final_answer = None

                candidate_letter = _extract_candidate_letter(final_answer) if final_answer else None

                if not loop_detected and candidate_letter is not None:
                    correct = (candidate_letter == ref_ans)
                elif loop_detected:
                    correct = False
                else:
                    correct = None

                if correct is True:
                    n_correct += 1
                if loop_detected:
                    n_loop += 1

                rec = {
                    "question_id":      qid,
                    "category":         q.get("category", "math"),
                    "src":              q.get("src", ""),
                    "model":            model_key,
                    "mode":             mode,
                    "attempt":          attempt,
                    "temperature":      MODEL_OPTIONS.get("temperature"),
                    "repeat_penalty":   MODEL_OPTIONS.get("repeat_penalty"),
                    "num_ctx":          NUM_CTX_THINKOFF if think_off else NUM_CTX_THINKON,
                    "question":         q["question"],
                    "options":          q["options"],
                    "reference_answer": ref_ans,
                    "answer_index":     ref_idx,
                    "loop_detected":    loop_detected,
                    "think_body":       think_body,
                    "final_answer":     final_answer,
                    "candidate_letter": candidate_letter,
                    "correct":          correct,
                    "elapsed_s":        elapsed,
                }
                append_result(out_path, rec)
                done.add(key)

                if loop_detected:
                    status = "LOOP "
                elif correct is True:
                    status = "✓    "
                elif correct is False:
                    status = "✗    "
                else:
                    status = "?box "

                think_len = len(think_body)
                print(f"    {status}  got={candidate_letter}  ref={ref_ans}  "
                      f"think={think_len:5d}c  elapsed={elapsed}s", flush=True)
                n_done += 1

    print(f"\n[DONE] generated={n_done}  correct={n_correct}  loops={n_loop}  errors={n_errors}")
    print(f"[OUT]  {out_path}")

    if os.path.exists(out_path):
        all_records = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
        if all_records:
            print_summary(all_records, category=category)


if __name__ == "__main__":
    main()
