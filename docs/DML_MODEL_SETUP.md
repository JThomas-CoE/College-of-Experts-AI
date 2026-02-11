# DML Model Setup and VRAM Configuration Guide

## Overview

This guide covers the DirectML (DML) quantized model setup for the College of Experts system. DML models provide efficient inference on AMD GPUs and Windows systems with significantly reduced VRAM requirements compared to full-precision models.

## Available DML Models

The following DML quantized models are available in `models/`:

| Model | Path | Context Length | Primary Use Case |
|-------|------|----------------|------------------|
| DeepSeek-R1-Distill-Qwen-7B-DML | `models/DeepSeek-R1-Distill-Qwen-7B-DML` | 32,768 tokens | General reasoning, fallback expert |
| Qwen2.5-Coder-7B-DML | `models/Qwen2.5-Coder-7B-DML` | 32,768 tokens | Code generation (Python, SQL DDL) |
| Qwen2.5-Math-7B-DML | `models/Qwen2.5-Math-7B-DML` | 4,096 tokens | Mathematical reasoning, proofs |
| BioMistral-7B-DML | `models/BioMistral-7B-DML` | 32,768 tokens | Medical/clinical applications |
| sqlcoder-7b-2-DML | `models/sqlcoder-7b-2-DML` | 16,384 tokens | Text-to-SQL queries (NOT schema design) |
| law-LLM-DML | `models/law-LLM-DML` | 2,048 tokens | Legal text (limited context) |

## VRAM Requirements

### Model Weights Memory Footprint

DML quantized models use INT4/INT8 quantization, resulting in much smaller memory footprints:

| Model | Approximate Weights Size | KV Cache (8K context) | Total Runtime |
|-------|-------------------------|----------------------|---------------|
| DeepSeek-R1-Distill-Qwen-7B-DML | ~5.0 GB | ~2.5 GB | ~8.0 GB |
| Qwen2.5-Coder-7B-DML | ~5.0 GB | ~2.5 GB | ~8.0 GB |
| Qwen2.5-Math-7B-DML | ~5.0 GB | ~2.5 GB | ~8.0 GB |
| BioMistral-7B-DML | ~5.0 GB | ~2.5 GB | ~8.0 GB |
| sqlcoder-7b-2-DML | ~5.0 GB | ~2.5 GB | ~8.0 GB |
| law-LLM-DML | ~5.0 GB | ~2.5 GB | ~8.0 GB |

**Note:** DeepSeek uses 32K context with efficient GQA. VRAM scheduler counts DeepSeek as 2 slots due to larger model architecture (28 layers, 3584 hidden size).

### Fixed Overhead

Always resident in VRAM:
- **BGE-M3 Embedding Model**: ~1.0 GB
- **System/Driver Overhead**: ~1.0 GB
- **Total Fixed**: ~2.0 GB

### Hardware Profile Recommendations

Configure the appropriate hardware profile in `config/hardware_profiles.yaml`:

#### Minimal (8GB VRAM)
```yaml
vram_budget_mb: 8000
max_concurrent_slots: 1
context_length: 4096
```
**Note:** With 8GB, you can load 1 model at a time with reduced context. Expect model swapping delays.

#### Compact (16GB VRAM)
```yaml
vram_budget_mb: 16000
max_concurrent_slots: 1
context_length: 8192
```
**Note:** Can preload next model while current executes. Sequential execution with smooth transitions.

#### Standard (32GB VRAM) - RECOMMENDED
```yaml
vram_budget_mb: 32000
max_concurrent_slots: 2
context_length: 8192
```
**Note:** Can run 2 experts in parallel. Good for most multi-domain queries.

#### Performance (48GB VRAM)
```yaml
vram_budget_mb: 48000
max_concurrent_slots: 4
context_length: 16384
```
**Note:** High parallelism. Can run 3 regular experts + 1 DeepSeek simultaneously.

#### Unlimited (64GB+ VRAM)
```yaml
vram_budget_mb: 64000
max_concurrent_slots: 6
context_length: 16384
```
**Note:** All savants can remain resident. No swapping overhead.

## Configuration

### SAVANT_MODELS Configuration

The model paths are configured in `demo_v12_full.py`:

```python
SAVANT_MODELS = {
    "python_backend": "models/Qwen2.5-Coder-7B-DML",
    "sql_schema_architect": "models/Qwen2.5-Coder-7B-DML",
    "html_css_specialist": "models/Qwen2.5-Coder-7B-DML",
    "security_architect": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
    "legal_contracts": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
    "math_expert": "models/Qwen2.5-Math-7B-DML",
    "medical_expert": "models/BioMistral-7B-DML",
    "architecture_expert": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
    "general_reasoner": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
}
```

### CUSTOM_VRAM_CONFIG

Adjust the VRAM configuration in `demo_v12_full.py`:

```python
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 48000,       # Total VRAM budget
    "max_expert_slots": 3,          # Regular expert slots
    "deepseek_slot_cost": 2,        # DeepSeek counts as 2 slots
    "embedding_mb": 2000,           # BGE-M3 embedding model
    "base_model_mb": 5000,          # DML model weights (~5GB)
    "deepseek_base_mb": 5000,       # DeepSeek base size
}
```

## DeepSeek DML Model Details

The DeepSeek-R1-Distill-Qwen-7B-DML model configuration (`genai_config.json`):

```json
{
    "model": {
        "context_length": 131072,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "type": "qwen2"
    },
    "search": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50
    }
}
```

### Key Features:
- **Architecture**: Qwen2-based with 28 layers
- **Context**: 131K tokens (longest of all models)
- **Attention**: Grouped Query Attention (GQA) with 4 KV heads
- **Provider**: DirectML execution provider
- **Primary Use**: General reasoning, complex fallbacks, security analysis

## VRAM Scheduler Behavior

### Slot Cost Calculation

The VRAM-aware scheduler (`src/vram_manager.py`) calculates slot costs as:

```
total_slot_vram = model_weights + kv_cache + activation_memory

where:
- model_weights = 5000 MB (DML quantized)
- kv_cache = tokens × layers × hidden_size × 2 (K+V) × 2 bytes (FP16)
- activation_memory = 500 MB (working memory)
```

### KV Cache Formula

```python
kv_bytes_per_token = 2 * num_layers * hidden_size * 2  # FP16
kv_cache_mb = (tokens × kv_bytes_per_token) / (1024 × 1024)
```

For DeepSeek (28 layers, 3584 hidden):
- Per token: ~400 bytes
- 8K context: ~3.2 GB
- 16K context: ~6.4 GB

### Pressure Thresholds

The scheduler responds to VRAM pressure levels:

| Pressure Level | Threshold | Behavior |
|----------------|-----------|----------|
| Normal | < 80% | Business as usual, preloading enabled |
| Elevated | 80-90% | Pause preloading |
| High | 90-95% | Warn, no new slots |
| Critical | > 95% | Emergency eviction of largest consumer |

## Model Loading Flow

1. **Request**: Expert requests model for slot execution
2. **Check**: Scheduler checks if model already loaded
3. **Reserve**: VRAM budget reserved (model + estimated KV cache)
4. **Load**: OGA backend loads model via DirectML
5. **Execute**: Generation with KV cache tracking
6. **Release**: VRAM reservation released, model may stay resident

## Troubleshooting

### Out of VRAM Errors

If you encounter VRAM exhaustion:

1. **Reduce context length** in hardware profile
2. **Reduce max_concurrent_slots** to 1
3. **Use minimal profile** for 8GB GPUs
4. **Check for model duplicates** - same model loaded under different names

### DirectML Not Available

If DirectML provider is not detected:

1. Ensure Windows 10/11 with updated drivers
2. Install DirectML-enabled ONNX Runtime:
   ```bash
   pip install onnxruntime-directml
   ```
3. Verify AMD GPU drivers are up to date

### Slow Model Loading

If models load slowly:

1. Models are loaded from disk on first use
2. Subsequent uses reuse resident models
3. Consider using `unlimited` profile to keep all models resident

## Performance Tips

1. **Use model reuse**: Multiple experts can share the same underlying model
2. **Preload enabled**: Enable preloading in hardware profile for smoother transitions
3. **Match profile to hardware**: Don't use `unlimited` profile if you don't have the VRAM
4. **Monitor pressure**: Watch VRAM pressure indicators in logs

## See Also

- `config/hardware_profiles.yaml` - Hardware profile definitions
- `src/vram_manager.py` - VRAM scheduling implementation
- `src/backends/oga_backend.py` - OGA/DirectML backend
- `docs/DML_MODEL_REFERENCE.md` - Quick reference for all DML models