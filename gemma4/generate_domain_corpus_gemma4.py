"""
generate_domain_corpus_gemma4.py
=================================
Generates domain corpus JSONL files by submitting expert-authored questions
to gemma4:26b-a4b-it-q8_0 (thinking ON) via the Ollama HTTP API.

For each question, the model is run with thinking enabled.  The response is
split into think block and answer:

  <think> ... </think>   →  think_generated.jsonl  (full think incl. closing tag)
  remaining text         →  qa_generated.jsonl      (question + answer stripped of think)

Output files (written incrementally — resumable):
  data/<domain>_qa_generated.jsonl
      {"question": "...", "answer": "...", "subfield": "...", "idx": N}

  data/<domain>_think_generated.jsonl
      {"text": "<full think block including </think>>", "subfield": "...", "idx": N}

Usage:
  python generate_domain_corpus_gemma4.py --domain physics
  python generate_domain_corpus_gemma4.py --domain engineering
  python generate_domain_corpus_gemma4.py --domain physics --smoke-test   # first 2 only
  python generate_domain_corpus_gemma4.py --domain physics --start-idx 5  # resume from idx
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")

OLLAMA_CHAT  = "http://localhost:11434/api/chat"
MODEL        = "gemma4:26b-a4b-it-q8_0"

# Generation parameters
NUM_CTX      = 32768   # context window — system + question + full think trace + answer
NUM_PREDICT  = 12288   # hard derivations can think 4-6k tokens; leave room for full answer
TEMPERATURE  = 0.6
TOP_K        = 64
TOP_P        = 0.95

DOMAIN_FILES = {
    "physics":     "physics_questions.jsonl",
    "engineering": "engineering_questions.jsonl",
}

# Extra fields in the questions JSONL that are passed through verbatim to the
# output records (e.g. "category": "recall|derivation|applied").
# Callers can add any fields they like; only these are explicitly allowed.
PASSTHROUGH_FIELDS = {"category"}

SYSTEM_PROMPT = (
    "You are an expert scientist and engineer with deep graduate-level knowledge. "
    "When asked a question, think through it carefully and provide a thorough, "
    "technically accurate answer with derivations, equations, and physical reasoning. "
    "Use precise scientific notation and terminology."
)

# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

# Closed think-block injected at the start of every message when --think-off is used.
# Forces thinking_budget=0 at the prompt level, complementing the think=False API flag.
THINK_OFF_PREFIX = "<think></think>\n"


def ollama_generate(question: str, model: str = MODEL, timeout: int = 600,
                    think_off: bool = False) -> str:
    """
    Call Ollama /api/chat.

    think_off=False (default):
        think=True in payload; model emits <think>...</think> via message.thinking.
        Returns reconstructed "<think>...</think>\nanswer" for split_think_answer().

    think_off=True:
        think=False in payload AND a closed <think></think> block is prepended to the
        question to force thinking_budget=0 at the prompt level (belt-and-suspenders).
        Returns answer string directly; think block is empty.
    """
    user_content = (THINK_OFF_PREFIX + question) if think_off else question
    payload = {
        "model":    model,
        "stream":   False,
        "think":    not think_off,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "options": {
            "temperature":  TEMPERATURE,
            "top_k":        TOP_K,
            "top_p":        TOP_P,
            "num_ctx":      NUM_CTX,
            "num_predict":  NUM_PREDICT,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        OLLAMA_CHAT,
        data    = data,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        result = json.loads(body)
        msg    = result["message"]
        if think_off:
            # think=False: content is the direct answer, thinking field absent/empty
            return msg.get("content", "").strip()
        else:
            think  = msg.get("thinking", "")   # Ollama puts think content here when think=True
            answer = msg.get("content", "")    # final answer after think completes
            if think:
                return f"<think>{think}</think>\n{answer}"
            return answer
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama response not valid JSON: {e}") from e


# ---------------------------------------------------------------------------
# Think/answer splitter
# ---------------------------------------------------------------------------

THINK_OPEN_RE  = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def split_think_answer(raw: str) -> tuple[str, str]:
    """
    Split raw model output into (think_block, answer).

    think_block includes the full <think>...</think> text including the
    closing </think> tag — those tokens are what we want in the think corpus.

    answer is everything after </think>, stripped of leading whitespace.
    If no <think> block is found, think_block is empty and answer is raw.
    """
    m_open  = THINK_OPEN_RE.search(raw)
    m_close = THINK_CLOSE_RE.search(raw)

    if m_open and m_close and m_close.end() > m_open.start():
        think_block = raw[m_open.start() : m_close.end()]  # includes both tags
        answer      = raw[m_close.end():].lstrip()
    elif m_open and not m_close:
        # Think block opened but not closed — treat everything as think
        think_block = raw[m_open.start():]
        answer      = ""
    else:
        think_block = ""
        answer      = raw.strip()

    return think_block, answer


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_done_indices(path: str) -> set[int]:
    """Return set of idx values already written to a JSONL file."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["idx"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate domain corpus via Ollama.")
    parser.add_argument("--domain",     required=True,  choices=list(DOMAIN_FILES.keys()),
                        help="Domain to generate: physics or engineering")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Only process first 2 questions")
    parser.add_argument("--start-idx",  type=int, default=0,
                        help="Skip questions with idx < start-idx (manual resume override)")
    parser.add_argument("--model",      default=MODEL,
                        help=f"Ollama model tag (default: {MODEL})")
    parser.add_argument("--questions-file", default="",
                        help="Override the default questions file for this domain. "
                             "Must not be the corpus question file — keeps provenance clean.")
    parser.add_argument("--out-prefix", default="",
                        help="Output filename prefix (default: same as --domain). "
                             "Use e.g. 'physics_bench' to write physics_bench_qa_generated.jsonl "
                             "without touching the profiling-corpus outputs.")
    parser.add_argument("--think-off", action="store_true",
                        help="Disable thinking: sets think=False in Ollama payload AND injects "
                             "a closed <think></think> block before the question. "
                             "Output prefix gets '_thinkoff' appended automatically unless "
                             "--out-prefix is specified explicitly.")
    args = parser.parse_args()

    model = args.model

    # Determine question source — --questions-file overrides domain default.
    # Never allow the override to silently point at the corpus source file.
    if args.questions_file:
        questions_path = os.path.join(DATA_DIR, args.questions_file) \
            if not os.path.isabs(args.questions_file) else args.questions_file
        corpus_default = os.path.join(DATA_DIR, DOMAIN_FILES[args.domain])
        if os.path.abspath(questions_path) == os.path.abspath(corpus_default):
            sys.exit("[ERROR] --questions-file must not point at the corpus source file. "
                     "Use a separate bench questions file to preserve provenance.")
    else:
        questions_path = os.path.join(DATA_DIR, DOMAIN_FILES[args.domain])

    if not os.path.exists(questions_path):
        sys.exit(f"[ERROR] Questions file not found: {questions_path}")

    # Output prefix — defaults to domain name, with _thinkoff suffix when --think-off set
    # and the caller has not specified an explicit prefix.
    if args.out_prefix:
        out_prefix = args.out_prefix
    elif args.think_off:
        out_prefix = args.domain + "_thinkoff"
    else:
        out_prefix = args.domain

    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]

    if args.smoke_test:
        questions = questions[:2]
        print(f"[SMOKE TEST] Running first 2 questions only.")

    out_qa    = os.path.join(DATA_DIR, f"{out_prefix}_qa_generated.jsonl")
    out_think = os.path.join(DATA_DIR, f"{out_prefix}_think_generated.jsonl")

    # Determine already-completed indices (cross-check both outputs)
    done_qa    = load_done_indices(out_qa)
    done_think = load_done_indices(out_think)
    done       = done_qa & done_think  # only skip if both written

    print(f"[INFO] Domain      : {args.domain}")
    print(f"[INFO] Model       : {model}")
    print(f"[INFO] Questions   : {len(questions)}  ({questions_path})")
    print(f"[INFO] Out prefix  : {out_prefix}")
    print(f"[INFO] Already done: {len(done)}")
    print(f"[INFO] Think mode  : {'OFF (think=False + closed-block injection)' if args.think_off else 'ON (think=True)'}")
    print(f"[INFO] QA output   : {out_qa}")
    print(f"[INFO] Think output: {out_think}")
    print()

    n_generated = 0
    n_skipped   = 0
    n_errors    = 0

    for idx, rec in enumerate(questions):
        if idx < args.start_idx:
            n_skipped += 1
            continue
        if idx in done:
            print(f"  [{idx+1:3d}/{len(questions)}] SKIP (already done): {rec['subfield']}")
            n_skipped += 1
            continue

        question  = rec["question"]
        subfield  = rec["subfield"]
        preview   = question[:80].replace("\n", " ")
        print(f"  [{idx+1:3d}/{len(questions)}] {subfield}: {preview}...")

        t0 = time.time()
        try:
            raw = ollama_generate(question, model=model, think_off=args.think_off)
        except RuntimeError as e:
            print(f"    ERROR: {e}")
            n_errors += 1
            continue
        elapsed = time.time() - t0

        think_block, answer = split_think_answer(raw)

        if not answer.strip():
            print(f"    WARN: Empty answer after think split — raw length={len(raw)} chars")

        # Write QA record — pass through any extra fields (e.g. category)
        qa_rec = {
            "question": question,
            "answer":   answer,
            "subfield": subfield,
            "idx":      idx,
        }
        for field in PASSTHROUGH_FIELDS:
            if field in rec:
                qa_rec[field] = rec[field]
        append_jsonl(out_qa, qa_rec)

        # Write think record (always — even if empty, for index consistency)
        think_rec = {
            "text":     think_block,
            "subfield": subfield,
            "idx":      idx,
        }
        for field in PASSTHROUGH_FIELDS:
            if field in rec:
                think_rec[field] = rec[field]
        append_jsonl(out_think, think_rec)

        think_len  = len(think_block)
        answer_len = len(answer)
        print(f"    OK  think={think_len:5d} chars  answer={answer_len:5d} chars  "
              f"elapsed={elapsed:.1f}s")
        n_generated += 1

    print()
    print(f"[DONE] generated={n_generated}  skipped={n_skipped}  errors={n_errors}")
    if n_errors:
        print(f"[WARN] {n_errors} errors — re-run to retry failed questions")


if __name__ == "__main__":
    main()
