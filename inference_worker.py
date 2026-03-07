#!/usr/bin/env python3
"""Single-shot inference worker for process-isolated OGA generation."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict


_THINKING_MODEL_MARKERS = ("nanbeige", "deepseek-r1", "qwq", "-r1")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _is_thinking_model(path: str) -> bool:
    p = (path or "").lower()
    return any(marker in p for marker in _THINKING_MODEL_MARKERS)


def _strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text or "")
    open_idx = text.lower().find("<think>")
    if open_idx != -1:
        text = text[:open_idx]
    return text


def _format_prompt(model_path: str, prompt: str, system: str = "") -> str:
    p = (model_path or "").lower()
    if "sqlcoder" in p:
        return (
            "### Task\nGenerate a SQL query to answer the following question.\n\n"
            f"### Instructions\n{prompt}\n\n### Response\n"
        )
    if "law" in p:
        sys_part = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"
    if "biomistral" in p or "bio" in p:
        sys_part = f"{system}\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"
    if system:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _read_ctx_limit(model_path: str) -> int:
    cfg_file = Path(model_path) / "genai_config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
            return int(cfg.get("model", {}).get("context_length", 4096))
        except Exception:
            pass
    return 4096


def _generate_once(request: Dict[str, Any]) -> str:
    import onnxruntime_genai as og

    model_path = request["model_path"]
    prompt = request.get("prompt", "")
    system = request.get("system", "")
    max_tokens = int(request.get("max_tokens", 1024))
    temperature = max(float(request.get("temperature", 0.2)), 0.01)
    think_budget = int(request.get("think_budget", 700))
    runtime_cap = int(request.get("max_runtime_seq_tokens", 4096))

    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    formatted = _format_prompt(model_path, prompt, system)
    input_tokens = tokenizer.encode(formatted)

    ctx_limit = min(_read_ctx_limit(model_path), runtime_cap)
    if len(input_tokens) > ctx_limit - 32:
        input_tokens = input_tokens[-(ctx_limit - 32):]

    if _is_thinking_model(model_path):
        initial_headroom = ctx_limit - len(input_tokens) - 1
        if initial_headroom <= 0:
            return ""

        reserve_for_answer = min(max_tokens, max(256, min(1024, initial_headroom // 3)))
        think_limit = min(think_budget, max(initial_headroom - reserve_for_answer - 1, 0))
        if think_limit < 1:
            think_tokens = []
            think_text = ""
            found_close = True
        else:
            p1 = og.GeneratorParams(model)
            p1.set_search_options(
                max_length=min(len(input_tokens) + think_limit, ctx_limit),
                temperature=temperature,
            )
            g1 = og.Generator(model, p1)
            g1.append_tokens(input_tokens)

            think_tokens = []
            found_close = False
            while not g1.is_done() and len(think_tokens) < think_limit:
                g1.generate_next_token()
                think_tokens.append(g1.get_next_tokens()[0])
                if len(think_tokens) % 4 == 0:
                    if "</think>" in tokenizer.decode(think_tokens).lower():
                        found_close = True
                        break

            think_text = tokenizer.decode(think_tokens)
            del g1
            del p1

        if not found_close:
            think_text = think_text.rstrip() + "\n</think>\n"

        phase2_text = formatted + think_text
        phase2_tokens = tokenizer.encode(phase2_text)
        if len(phase2_tokens) > ctx_limit - 32:
            phase2_tokens = phase2_tokens[-(ctx_limit - 32):]
        answer_budget = min(max_tokens, ctx_limit - len(phase2_tokens) - 10)
        if answer_budget <= 0:
            return ""

        p2 = og.GeneratorParams(model)
        p2.set_search_options(
            max_length=min(len(phase2_tokens) + answer_budget, ctx_limit),
            temperature=temperature,
        )
        g2 = og.Generator(model, p2)
        g2.append_tokens(phase2_tokens)

        answer_tokens = []
        while not g2.is_done() and len(answer_tokens) < answer_budget:
            g2.generate_next_token()
            answer_tokens.append(g2.get_next_tokens()[0])

        response = tokenizer.decode(answer_tokens)
        del g2
        del p2
        return _strip_think(response).strip()

    available_gen = min(max_tokens, ctx_limit - len(input_tokens) - 10)
    if available_gen <= 0:
        return ""

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=min(len(input_tokens) + available_gen, ctx_limit),
        temperature=temperature,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    output_tokens = []
    while not generator.is_done() and len(output_tokens) < available_gen:
        generator.generate_next_token()
        output_tokens.append(generator.get_next_tokens()[0])

    response = tokenizer.decode(output_tokens)
    del generator
    del params
    return response.strip()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: inference_worker.py <request.json> <response.json>", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])

    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        t0 = time.time()
        result = _generate_once(request)
        payload = {
            "ok": True,
            "result": result,
            "elapsed_s": round(time.time() - t0, 3),
            "model": Path(request.get("model_path", "")).name,
        }
        response_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            response_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
