import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
print("Importing backend...")
from src.backends.oga_backend import OGABackend
print("Done importing.")
backend = OGABackend("models/Qwen2.5-Coder-7B-DML")
print("Model loaded.")
