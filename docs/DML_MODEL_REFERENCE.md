# DML Model Quick Reference

## Model Overview

All DML models use INT4/INT8 quantization and run via DirectML on AMD GPUs and Windows systems.

## Model Specifications

### DeepSeek-R1-Distill-Qwen-7B-DML
```yaml
Path: models/DeepSeek-R1-Distill-Qwen-7B-DML
Context: 131,072 tokens
Weights: ~5.0 GB
KV Cache (8K): ~3.0 GB
Total Runtime: ~9.0 GB
Slot Cost: 2x (larger KV cache)

Architecture:
  Type: qwen2
  Layers: 28
  Hidden Size: 3584
  Attention Heads: 28
  KV Heads: 4 (GQA)

Use Cases:
  - General reasoning and analysis
  - Fallback for complex tasks
  - Security architecture guidance
  - Legal document drafting
  - Multi-step reasoning

Config: genai_config.json
  Temperature: 0.6
  Top-P: 0.95
  Top-K: 50
```

### Qwen2.5-Coder-7B-DML
```yaml
Path: models/Qwen2.5-Coder-7B-DML
Context: 32,768 tokens
Weights: ~5.0 GB
KV Cache (8K): ~2.5 GB
Total Runtime: ~8.0 GB
Slot Cost: 1x

Use Cases:
  - Python backend development
  - SQL DDL/schema design
  - HTML/CSS/JavaScript generation
  - FastAPI/Flask/Django code
  - API endpoint implementation

Notes:
  - Best general-purpose coding model
  - Handles multiple languages well
  - Good for structured output
```

### Qwen2.5-Math-7B-DML
```yaml
Path: models/Qwen2.5-Math-7B-DML
Context: 4,096 tokens
Weights: ~5.0 GB
KV Cache (4K): ~1.5 GB
Total Runtime: ~7.0 GB
Slot Cost: 1x

Use Cases:
  - Mathematical proofs
  - Calculations and equations
  - Optimization problems
  - Statistical analysis

Notes:
  - Smallest context window
  - Specialized for math reasoning
  - Use for math-specific tasks only
```

### BioMistral-7B-DML
```yaml
Path: models/BioMistral-7B-DML
Context: 32,768 tokens
Weights: ~5.0 GB
KV Cache (8K): ~2.5 GB
Total Runtime: ~8.0 GB
Slot Cost: 1x

Use Cases:
  - Medical/clinical applications
  - Biology and life sciences
  - Healthcare data processing
  - Medical documentation

Notes:
  - Specialized medical knowledge
  - Good for clinical contexts
```

### sqlcoder-7b-2-DML
```yaml
Path: models/sqlcoder-7b-2-DML
Context: 16,384 tokens
Weights: ~5.0 GB
KV Cache (8K): ~2.5 GB
Total Runtime: ~8.0 GB
Slot Cost: 1x

Use Cases:
  - Text-to-SQL query generation
  - SELECT statement writing
  - Query optimization suggestions

Notes:
  - NOT for schema design (use Coder)
  - Specialized for query generation
  - Good for natural language to SQL
```

### law-LLM-DML
```yaml
Path: models/law-LLM-DML
Context: 2,048 tokens
Weights: ~5.0 GB
KV Cache (2K): ~0.8 GB
Total Runtime: ~6.5 GB
Slot Cost: 1x

Use Cases:
  - Legal text generation
  - Contract clauses
  - Policy drafting

Notes:
  - Very limited context (2K)
  - Use DeepSeek for complex legal tasks
  - May truncate on longer documents
```

## VRAM Budget Calculator

### Formula
```
Total VRAM Needed = Fixed Overhead + (Model Slots × Slot Cost)

where:
  Fixed Overhead = 2 GB (embedding + system)
  Model Slot = 8 GB (weights + KV cache + activation)
  DeepSeek Slot = 16 GB (weights + large KV cache)
```

### Quick Calculator

| Setup | Regular Slots | DeepSeek Slots | Total VRAM Needed |
|-------|--------------|----------------|-------------------|
| 1 expert | 1 | 0 | ~10 GB |
| 1 expert + DeepSeek | 0 | 1 | ~18 GB |
| 2 experts | 2 | 0 | ~18 GB |
| 2 experts + DeepSeek | 1 | 1 | ~26 GB |
| 3 experts | 3 | 0 | ~26 GB |
| 3 experts + DeepSeek | 2 | 1 | ~34 GB |
| 4 experts | 4 | 0 | ~34 GB |

## Expert-to-Model Mapping

| Expert Role | Model | Slot Cost |
|-------------|-------|-----------|
| python_backend | Qwen2.5-Coder-7B-DML | 1 |
| sql_schema_architect | Qwen2.5-Coder-7B-DML | 1 |
| html_css_specialist | Qwen2.5-Coder-7B-DML | 1 |
| security_architect | DeepSeek-R1-Distill-Qwen-7B-DML | 2 |
| legal_contracts | DeepSeek-R1-Distill-Qwen-7B-DML | 2 |
| math_expert | Qwen2.5-Math-7B-DML | 1 |
| medical_expert | BioMistral-7B-DML | 1 |
| architecture_expert | DeepSeek-R1-Distill-Qwen-7B-DML | 2 |
| general_reasoner | DeepSeek-R1-Distill-Qwen-7B-DML | 2 |

## Model Reuse Strategy

Multiple experts can share the same underlying model:

```
Qwen2.5-Coder-7B-DML serves:
  - python_backend
  - sql_schema_architect  
  - html_css_specialist

DeepSeek-R1-Distill-Qwen-7B-DML serves:
  - security_architect
  - legal_contracts
  - architecture_expert
  - general_reasoner
```

This means a query using `python_backend` + `security_architect` only needs:
- 1 slot for Coder (shared by all code experts)
- 2 slots for DeepSeek (security)
- Total: 3 slots (~26 GB)

## Context Length Guidelines

### Maximum Input + Output by Model

| Model | Context | Recommended Max Input | Recommended Max Output |
|-------|---------|----------------------|------------------------|
| DeepSeek | 131K | 100K | 20K |
| Coder | 32K | 24K | 6K |
| BioMistral | 32K | 24K | 6K |
| SQLCoder | 16K | 12K | 3K |
| Math | 4K | 3K | 1K |
| Law | 2K | 1.5K | 0.5K |

### Context Scaling Impact on VRAM

| Context | KV Cache Size | Additional VRAM |
|---------|---------------|-----------------|
| 4K | 1.5 GB | baseline |
| 8K | 3.0 GB | +1.5 GB |
| 16K | 6.0 GB | +4.5 GB |
| 32K | 12.0 GB | +10.5 GB |

**Recommendation**: Use the hardware profile's `context_length` setting to limit maximum context and control VRAM usage.

## Performance Characteristics

### Generation Speed (approximate, varies by GPU)

| Model | Tokens/Second | Relative Speed |
|-------|---------------|----------------|
| DeepSeek | 15-25 | 1.0x (baseline) |
| Coder | 20-30 | 1.2x |
| Math | 25-35 | 1.4x |
| BioMistral | 20-30 | 1.2x |
| SQLCoder | 20-30 | 1.2x |
| Law | 25-35 | 1.4x |

### Model Loading Time

| Storage | First Load | Subsequent Uses |
|---------|-----------|-----------------|
| NVMe SSD | 2-4s | 0s (resident) |
| SATA SSD | 4-8s | 0s (resident) |
| HDD | 10-20s | 0s (resident) |

## Troubleshooting Matrix

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| OOM on load | Insufficient VRAM | Reduce `max_concurrent_slots` |
| OOM during gen | KV cache explosion | Reduce `context_length` |
| Slow generation | High VRAM pressure | Close other GPU apps |
| Model not found | Wrong path | Check `SAVANT_MODELS` paths |
| DML not available | Missing DirectML | Install `onnxruntime-directml` |
| Truncated output | Context exceeded | Reduce input or output tokens |

## Configuration Quick Reference

### Minimal Setup (8GB VRAM)
```python
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 8000,
    "max_expert_slots": 1,
    "deepseek_slot_cost": 2,
    "embedding_mb": 2000,
    "base_model_mb": 5000,
    "deepseek_base_mb": 5000,
}
```

### Standard Setup (32GB VRAM)
```python
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 32000,
    "max_expert_slots": 2,
    "deepseek_slot_cost": 2,
    "embedding_mb": 2000,
    "base_model_mb": 5000,
    "deepseek_base_mb": 5000,
}
```

### Performance Setup (48GB VRAM)
```python
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 48000,
    "max_expert_slots": 3,
    "deepseek_slot_cost": 2,
    "embedding_mb": 2000,
    "base_model_mb": 5000,
    "deepseek_base_mb": 5000,
}