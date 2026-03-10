# `coe_demo/backends/` — Pluggable Inference Backends (Planned)

This directory is reserved for swappable inference-backend modules.  It is
intentionally empty in v1.5; the two active backends are implemented inline
in `models.py`.

## Planned layout

| File | Description |
|------|-------------|
| `ollama_backend.py` | Ollama HTTP API — specialist MoE models (CoE-python2, CoE-WEB2) |
| `oga_backend.py` | ONNX Runtime GenAI — ONNX Supervisor model (Nanbeige4.1-3B-INT4) |
| `vllm_backend.py` | vLLM server backend (future cloud/server deployment) |

## Extraction roadmap

Once the backends are extracted from `models.py` each module will expose a
single `generate(prompt, options) -> str` callable so the pipeline layer
(`pipeline.py`) can swap providers without touching inference logic.

This aligns v1.5 with the parent-repo `src/` architecture, where the
`LocalKnowledgeBase`, `MemoryBackbone`, and `EmbeddingManager` are already
decoupled from the demo entry-point.
