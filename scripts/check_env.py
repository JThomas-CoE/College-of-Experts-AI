"""Quick environment check for College of Experts."""
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

import transformers
print(f"Transformers: {transformers.__version__}")

# Check our modules
try:
    from src import Router, ExpertLoader, SharedLatentSpace
    print("Core imports: OK")
except Exception as e:
    print(f"Core imports: FAILED - {e}")

try:
    from src.experts import ALL_EXPERT_PERSONAS, get_tool_executor
    print(f"Experts: {len(ALL_EXPERT_PERSONAS)}")
    print(f"Tools: {get_tool_executor().list_available()}")
except Exception as e:
    print(f"Experts: FAILED - {e}")

try:
    from src.backends import TransformersBackend
    backend = TransformersBackend(device="auto")
    print(f"Backend device: {backend._detect_best_device()}")
except Exception as e:
    print(f"Backend: FAILED - {e}")

print("\n=== Environment Ready ===")
