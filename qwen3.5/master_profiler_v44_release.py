"""
master_profiler_v44_release.py  —  CoE histogram collection profiler (release copy)

This is the GitHub release version of master_profiler_v44.py. Machine-specific
paths have been replaced with paths relative to this script's location, and the
vision augmentation catalog path (not distributed) is set to None.

BEFORE RUNNING:
  1. Set MODEL_PATH below to your local Qwen3.5-35B-A3B safetensors directory.
     The model is not included in this repository (~70 GB BF16).
     Download: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

  2. conda activate <your-env>      # see environment.yml
     python master_profiler_v44_release.py

NOTE: This script is architecture-specific to Qwen3.5-35B-A3B (40 layers,
256 routed experts/layer, BF16 safetensors shards). It will not work with other
model architectures without modification to the hook registration and shard-loading
logic.

NOTE: Visual domain profiling (CATALOG_PATH) requires a vision augmentation
catalog file that is not included in this repository. Pre-computed visual
histograms are already included in histograms/final/ — visual profiling passes
are silently skipped when CATALOG_PATH is None.
"""
import os
import torch
import sys
import gc
import json
import re
import psutil
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from PIL import Image

# Force unbuffered output for logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- CPU PARALLELISM ---
torch.set_num_threads(28)
print(f"Parallel threads set to: {torch.get_num_threads()}")

# --- Configuration Paths ---
_HERE = os.path.dirname(os.path.abspath(__file__))

# ← SET THIS: absolute path to your local Qwen3.5-35B-A3B safetensors directory
MODEL_PATH         = r"<path-to-Qwen3.5-35B-A3B>"

OUTPUT_DIR         = os.path.join(_HERE, "histograms", "final")
TEXT_DATA_ROOT     = os.path.join(_HERE, "data", "curated")
HYDRATED_DATA_ROOT = os.path.join(_HERE, "data", "curated", "hydrated")
STATUS_FILE        = os.path.join(_HERE, "profiling_status.json")

# Vision augmentation catalog — not included in this repository.
# Set to a valid path if you have a vision_augmentation_catalog.md file;
# visual domain profiling is skipped automatically when this is None.
CATALOG_PATH = None

NUM_LAYERS = 40
NUM_EXPERTS = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_mem_usage(label=""):
    ram = psutil.virtual_memory()
    print(f"[{label}] RAM Used: {ram.used/1024**3:.2f}GB / {ram.total/1024**3:.2f}GB")

# --- Model Loading Logic ---
print("--- MoE Master Profiler (v44 HYDRATION AWARE) ---")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

with init_empty_weights():
    model = Qwen3_5MoeForConditionalGeneration(config)

model_keys = set(dict(model.named_parameters()).keys())

def load_shards_to_ram():
    print_mem_usage("PRE-LOAD")
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(list(set(weight_map.values())))
    
    for shard_file in tqdm(shards, desc="Establishing RAM Residency"):
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for param_name in f.keys():
                if "mtp." in param_name: continue
                actual_name = param_name if param_name in model_keys else f"model.{param_name}"
                if actual_name not in model_keys: continue
                tensor = f.get_tensor(param_name).clone().contiguous()
                set_module_tensor_to_device(model, actual_name, device="cpu", value=tensor.to(torch.bfloat16))
        gc.collect()
    model.to(torch.bfloat16).cpu().eval()
    print_mem_usage("POST-LOAD")

# --- Global Tracking & Hooks ---
# NEW: 3D Histogram [Layer, Expert, Rank (0-7)]
current_histograms = torch.zeros((NUM_LAYERS, NUM_EXPERTS, 8), dtype=torch.int64)

def get_activation_hook(layer_idx):
    def hook(module, inp, out):
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            # indices shape: (tokens, 8)
            indices = out[2].detach()
            for rank_idx in range(8):
                # indices is 2D: [total_tokens, top_k]
                # we want all tokens for a specific rank
                rank_indices = indices[:, rank_idx]
                current_histograms[layer_idx, :, rank_idx] += torch.bincount(rank_indices, minlength=NUM_EXPERTS)
    return hook

for i, layer in enumerate(model.model.language_model.layers):
    layer.mlp.gate.register_forward_hook(get_activation_hook(i))

# --- Dataset Parsing ---
def get_catalog_visual_data(domain_marker):
    if CATALOG_PATH is None or not os.path.exists(CATALOG_PATH): return []
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = rf"{re.escape(domain_marker)}.*?(?=\n## [^#]|$)"
    section_match = re.search(pattern, content, re.S)
    if not section_match: return []
    section = section_match.group(0)
    entries = re.findall(r"!\[.*?\]\((.*?)\).*?\*\*Caption\(Answer\)\*\*: (.*?)(?=\n<!-- slide -->|\n`````|\n!\[|$)", section, re.S)
    data = [{"image": p.strip(), "caption": c.strip()} for p, c in entries]
    print(f" Found {len(data)} visual samples for {domain_marker}")
    return data

def get_jsonl_text_data(file_pattern, use_hydrated=True):
    data_out = []
    if use_hydrated and os.path.exists(HYDRATED_DATA_ROOT):
        hydrated_files = [f for f in os.listdir(HYDRATED_DATA_ROOT) if f.endswith(".jsonl") and any(p in f for p in file_pattern)]
        if hydrated_files:
            print(f" [Info] Using HYDRATED sources for {file_pattern}: {hydrated_files}")
            for f_name in hydrated_files:
                f_path = os.path.join(HYDRATED_DATA_ROOT, f_name)
                with open(f_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            for k in ["text", "prompt", "content", "instruction", "input", "question"]:
                                if entry.get(k):
                                    data_out.append({"text": str(entry[k]).strip()})
                                    break
                        except: continue
            return data_out
    files = [f for f in os.listdir(TEXT_DATA_ROOT) if f.endswith(".jsonl") and any(p in f for p in file_pattern)]
    for f_name in files:
        f_path = os.path.join(TEXT_DATA_ROOT, f_name)
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    text = ""
                    for k in ["text", "prompt", "content", "instruction", "input", "question"]:
                        if entry.get(k):
                            text = str(entry[k]).strip()
                            break
                    if text: data_out.append({"text": text})
                except: continue
    return data_out

# --- Profiler Execution ---
def run_profile(data, label, is_visual=False):
    print(f"\n>>> Running Profile: {label} (Samples: {len(data)})")
    os.environ["TORCH_DISABLE_ADDR2LINE"] = "1"
    
    # Check for existing checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, f"{label}_checkpoint.pt")
    meta_path = os.path.join(OUTPUT_DIR, f"{label}_meta.json")
    start_idx = 0
    
    current_histograms.zero_()
    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        try:
            current_histograms.copy_(torch.load(ckpt_path))
            with open(meta_path, 'r') as f:
                start_idx = json.load(f).get("last_index", 0) + 1
            print(f" [RESUME] Found checkpoint at index {start_idx-1}. Resuming from {start_idx}...")
        except Exception as e:
            print(f" [WARN] Failed to load checkpoint: {e}. Starting from scratch.")
            current_histograms.zero_()
            start_idx = 0

    if start_idx >= len(data):
        print(f" [Finish] All samples for {label} already processed via checkpoint.")
        return

    for i in tqdm(range(start_idx, len(data)), desc=label):
        entry = data[i]
        try:
            if is_visual:
                image_path = entry["image"]
                text = entry["caption"]
                if not os.path.exists(image_path): continue
                image = Image.open(image_path).convert("RGB")
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            else:
                text = entry["text"]
                inputs = processor(text=[text], return_tensors="pt", padding=True)
            
            # Move inputs to CPU and ensure proper dtype
            final_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to("cpu")
                    if torch.is_floating_point(v):
                        v = v.to(torch.bfloat16)
                    final_inputs[k] = v
                else:
                    final_inputs[k] = v

            with torch.no_grad():
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    model(**final_inputs)
            
            interval = 5 if is_visual else 20
            if (i + 1) % interval == 0:
                torch.save(current_histograms.clone(), ckpt_path)
                with open(meta_path, 'w') as f:
                    json.dump({"last_index": i}, f)
                
        except Exception as e:
            print(f" [Err] {label} Sample {i} Error:")
            import traceback
            traceback.print_exc()
            continue

    # Final Save
    torch.save(current_histograms.clone(), os.path.join(OUTPUT_DIR, f"{label}_histogram.pt"))
    # Clear metadata on successful completion
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    if os.path.exists(meta_path): os.remove(meta_path)
    print(f" Saved {label} histogram. Grand Sum: {current_histograms.sum().item()}")

def update_status(label, state):
    status = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f: status = json.load(f)
    status[label] = state
    with open(STATUS_FILE, 'w') as f: json.dump(status, f, indent=4)

def check_status(label):
    if not os.path.exists(STATUS_FILE): return "pending"
    with open(STATUS_FILE, 'r') as f: return json.load(f).get(label, "pending")

# --- MAIN LOOP ---
DOMAIN_CONFIG = [
    {"id": "physics", "text_patterns": ["05_science_physics"], "vis_marker": "## Science: Physics"},
    {"id": "archeology", "text_patterns": ["12_archeology"], "vis_marker": "## Science: Archaeology"},
    {"id": "bio_chem", "text_patterns": ["06_science_bio_chem"], "vis_marker": "## Science: Bio-Chem"},
    {"id": "math", "text_patterns": ["07_science_adv_math", "07_science_math"], "vis_marker": "## Science: Visual Math"},
    {"id": "earth_science", "text_patterns": ["11_earth_sciences"], "vis_marker": "## Science: Earth Science"},
    {"id": "applied_engineering", "text_patterns": ["10_applied_engineering"], "vis_marker": None},
    {"id": "vocational_trades", "text_patterns": ["09_vocational_engineering", "10_tech_engineering"], "vis_marker": None},
    {"id": "coding_web", "text_patterns": ["01_coding_web"], "vis_marker": "## Computer Science"},
    {"id": "coding_systems", "text_patterns": ["02_coding_systems"], "vis_marker": None},
    {"id": "coding_os", "text_patterns": ["03_coding_os"], "vis_marker": None},
    {"id": "coding_sql", "text_patterns": ["04_coding_sql"], "vis_marker": None},
    {"id": "humanities", "text_patterns": ["08_humanities", "09_liberal_arts", "14_humanities"], "vis_marker": None},
]

load_shards_to_ram()

for domain in DOMAIN_CONFIG:
    dom_id = domain["id"]
    
    # Textual
    label_t = f"{dom_id}_textual"
    if check_status(label_t) != "complete":
        text_data = get_jsonl_text_data(domain["text_patterns"], use_hydrated=True)
        run_profile(text_data, label_t, is_visual=False)
        update_status(label_t, "complete")
    else: print(f" [Skip] {label_t} complete.")

    # Visual
    if domain["vis_marker"]:
        label_v = f"{dom_id}_visual"
        if check_status(label_v) != "complete":
            vis_data = get_catalog_visual_data(domain["vis_marker"])
            run_profile(vis_data, label_v, is_visual=True)
            update_status(label_v, "complete")
        else: print(f" [Skip] {label_v} complete.")

print("\n--- MASTER PROFILING COMPLETED ---")
print_mem_usage("FINAL")
