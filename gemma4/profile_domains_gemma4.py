"""
profile_domains_gemma4.py
==========================
Generic per-domain MoE activation profiler for Gemma4-26B-A4B.

Profiles any set of domain corpora specified on the command line.  Each domain
is identified by a key, a JSONL file path, and a TEXT_SPEC that controls how
text is extracted from each record.  Results are written to a single .pt file
keyed by domain ID, in the normalized format expected by build_coe_mask.py.

TEXT_SPEC forms
---------------
  prompt                           Single field name (with fallback chain)
  question+answer                  Join two fields with newline
  {context}\\n\\nQ: {question}        Python format_map template

Output (.pt) schema per domain
-------------------------------
  freq         (L, E, K)  float32   token-normalised selection frequency
  weight_mean  (L, E, K)  float32   mean softmax router weight
  histogram    (L, E, K)  int64     raw token counts (provenance)
  wsum         (L, E, K)  float32   raw weight sums (provenance)
  n_tokens     int                  total tokens processed
  n_samples    int                  total samples processed
  hist_fw      (L, E, K)  int64     framework-only pass (zeros if no fw data)
  wsum_fw      (L, E, K)  float32   framework-only weight sums
  n_tokens_fw  int                  framework-only token count

Framework pass
--------------
  If any record has a "framework" field in the JSONL, a second forward pass is
  run over those framework-only strings.  This is used for KB calibration
  corpora (structural baseline subtraction).  For plain-text corpora the fw
  tensors are stored as zeros without extra compute.

Usage
-----
  # Profile multiple coding language sub-corpora:
  python profile_domains_gemma4.py \\
    --output-pt  histograms/final/coding_results.pt \\
    --domain  coding_python  data/coding_python.jsonl  prompt \\
    --domain  coding_cpp     data/coding_cpp.jsonl     prompt \\
    --domain  coding_go      data/coding_go.jsonl      prompt \\
    --domain  coding_rust    data/coding_rust.jsonl    prompt

  # Re-run only specific domains from an existing run:
  python profile_domains_gemma4.py \\
    --output-pt  histograms/final/coding_results.pt \\
    --domain  coding_python  data/coding_python.jsonl  prompt \\
    --domain  coding_go      data/coding_go.jsonl      prompt \\
    --domains-only  coding_go

  # Template TEXT_SPEC (KB probe format):
  python profile_domains_gemma4.py \\
    --output-pt  histograms/final/probe_results.pt \\
    --domain  probe_naked     data/probe_coding.jsonl  question \\
    --domain  probe_matched   data/probe_coding.jsonl  "{matched_chunk}\\n\\nQ: {question}" \\
    --domain  probe_unmatched data/probe_coding.jsonl  "{unmatched_chunk}\\n\\nQ: {question}"

  # List configured domains and exit:
  python profile_domains_gemma4.py --output-pt x.pt --domain k p t --list-domains

Next step
---------
  python build_coe_mask.py \\
    --pt-file  histograms/final/coding_results.pt \\
    --domain-weights  coding_python:0.35,coding_cpp:0.25,coding_go:0.20,coding_rust:0.20 \\
    --output-name  coding \\
    --no-pass2 \\
    --budget 64
"""

import argparse
import gc
import json
import os
import sys
import torch
import psutil
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = r"C:\RyzenAI\college of experts\gemma4\gemma-4-26B-A4B-it"
DEFAULT_CHECKPOINT_EVERY = 20
DEFAULT_MAX_LENGTH = 1024

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Gemma4 generic per-domain MoE activation profiler",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--output-pt", required=True, metavar="PATH",
    help="Output .pt results file.  Status JSON is derived automatically "
         "by replacing .pt with _status.json.",
)
parser.add_argument(
    "--domain", nargs=3, action="append", metavar=("KEY", "PATH", "TEXT_SPEC"),
    required=True,
    help="Domain to profile.  Repeatable.  "
         "TEXT_SPEC: field name | field+field | {template}.",
)
parser.add_argument(
    "--domains-only", nargs="+", metavar="KEY",
    help="Run only these domain keys (must be listed in --domain).  "
         "Other keys are skipped even if not yet complete.",
)
parser.add_argument(
    "--list-domains", action="store_true",
    help="Print all --domain entries and exit without loading the model.",
)
parser.add_argument(
    "--model-path", default=DEFAULT_MODEL_PATH, metavar="PATH",
    help="Path to the Gemma4 model directory.",
)
parser.add_argument(
    "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY, metavar="N",
    help="Save incremental checkpoint every N samples.",
)
parser.add_argument(
    "--max-length", type=int, default=DEFAULT_MAX_LENGTH, metavar="N",
    help="Tokenizer max_length (truncation).",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build domain registry from CLI
# ---------------------------------------------------------------------------
domain_registry = {}   # KEY -> (PATH, TEXT_SPEC)
for key, path, text_spec in args.domain:
    if key in domain_registry:
        parser.error(f"Duplicate --domain key: {key}")
    domain_registry[key] = (path, text_spec)

if args.list_domains:
    print("Configured domains:")
    for key, (path, spec) in domain_registry.items():
        print(f"  {key:<35}  {path}  [{spec}]")
    sys.exit(0)

if args.domains_only:
    unknown = [k for k in args.domains_only if k not in domain_registry]
    if unknown:
        parser.error(f"--domains-only references unknown keys: {unknown}\n"
                     f"  Valid keys: {list(domain_registry.keys())}")
    active_domains = {k: domain_registry[k] for k in args.domains_only}
else:
    active_domains = dict(domain_registry)

# Derive paths
OUTPUT_PT   = os.path.abspath(args.output_pt)
STATUS_FILE = OUTPUT_PT.replace(".pt", "_status.json")
OUTPUT_DIR  = os.path.dirname(OUTPUT_PT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Architecture discovery (no model load yet)
# ---------------------------------------------------------------------------
def print_mem(label=""):
    ram = psutil.virtual_memory()
    print(f"[{label}] RAM {ram.used/1024**3:.2f} / {ram.total/1024**3:.2f} GB")

torch.set_num_threads(28)
print(f"Parallel threads: {torch.get_num_threads()}")
print("--- Gemma4 Domain Profiler ---")
print(f"Output PT   : {OUTPUT_PT}")
print(f"Status JSON : {STATUS_FILE}")
print(f"Active domains ({len(active_domains)}): {list(active_domains.keys())}")
print()

print("Loading config ...")
config   = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
text_cfg = config.text_config

NUM_LAYERS  = text_cfg.num_hidden_layers
NUM_EXPERTS = text_cfg.num_experts
TOP_K       = text_cfg.top_k_experts
print(f"Architecture: {NUM_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K} routing")

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
print_mem("PRE-LOAD")
print("Loading Gemma4 weights to CPU (bfloat16) ...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print_mem("POST-LOAD")

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------
current_histograms = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.int64)
current_weight_sum = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
current_hist_ew    = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
n_tokens_counter   = torch.zeros(1, dtype=torch.int64)

moe_layer_indices = []
handles = []

def make_router_hook(layer_idx):
    def hook(module, inp, out):
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            indices = out[2].detach()          # (T, TOP_K)
            weights = out[1].detach().float()  # (T, TOP_K)
            # Within-top-K spread: proxy for routing confidence after renormalisation.
            # High spread (w[0]≫w[-1]) → router was decisive; low spread → diffuse.
            # Clamped to ≥0 to guard against numerical noise.
            spread = (weights[:, 0] - weights[:, TOP_K - 1]).clamp(min=0)  # (T,)
            for rank_idx in range(TOP_K):
                rank_indices = indices[:, rank_idx]
                rank_weights = weights[:, rank_idx]
                current_histograms[layer_idx, :, rank_idx].add_(
                    torch.bincount(rank_indices, minlength=NUM_EXPERTS).to(torch.int64)
                )
                current_weight_sum[layer_idx, :, rank_idx].add_(
                    torch.zeros(NUM_EXPERTS, dtype=torch.float32)
                         .scatter_add_(0, rank_indices, rank_weights)
                )
                # EW accumulator: spread × w[rank] — up-weights decisive tokens
                ew_weights = spread * rank_weights
                current_hist_ew[layer_idx, :, rank_idx].add_(
                    torch.zeros(NUM_EXPERTS, dtype=torch.float32)
                         .scatter_add_(0, rank_indices, ew_weights)
                )
    return hook

text_model = model.model.language_model
for i, layer in enumerate(text_model.layers):
    if getattr(layer, "enable_moe_block", False):
        moe_layer_indices.append(i)
        h = layer.router.register_forward_hook(make_router_hook(i))
        handles.append(h)

print(f"MoE layers hooked: {len(moe_layer_indices)}  "
      f"indices {moe_layer_indices[:5]}...{moe_layer_indices[-5:] if len(moe_layer_indices) > 5 else ''}")

# ---------------------------------------------------------------------------
# TEXT_SPEC resolver
# ---------------------------------------------------------------------------
def resolve_text(spec, record):
    """
    Resolve a TEXT_SPEC against one JSONL record.

    Three modes:
      "{field} ..."  — Python format_map template
      "f1+f2+f3"    — join named fields with newline
      "field"        — single field, with fallback chain if field is empty/missing
    """
    try:
        if "{" in spec:
            text = spec.format_map(record)
        elif "+" in spec:
            parts = [str(record.get(f, "")).strip() for f in spec.split("+")]
            text = "\n".join(p for p in parts if p)
        else:
            # Single field: try the named field first, then fallback chain
            for field in [spec, "text", "content", "instruction", "input", "question"]:
                val = record.get(field)
                if val:
                    text = str(val).strip()
                    break
            else:
                text = ""
    except (KeyError, AttributeError):
        text = ""
    return text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_domain_data(path, text_spec):
    """
    Load samples from a JSONL file, applying text_spec to each record.

    Returns list of dicts with at minimum a "text" key.  Records that
    also have a "framework" field are passed through for the fw-calibration
    pass automatically.
    """
    data_out = []
    n_skip = 0

    if not os.path.exists(path):
        print(f"  [WARN] File not found: {path}")
        return data_out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = resolve_text(text_spec, record)
                if not text:
                    n_skip += 1
                    continue
                rec = {"text": text}
                # Preserve framework field for KB calibration corpora
                for extra in ("framework", "framework_id", "domain"):
                    if extra in record:
                        rec[extra] = record[extra]
                data_out.append(rec)
            except Exception:
                n_skip += 1
                continue

    print(f"  [data] {os.path.basename(path)}: {len(data_out)} samples loaded"
          + (f", {n_skip} skipped" if n_skip else ""))
    return data_out


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------
def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------
def run_forward_pass(text):
    """Tokenize and run one forward pass.  Returns token count or 0 on error."""
    inputs = processor.tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=args.max_length,
    )
    n_tok = inputs["input_ids"].shape[-1]
    n_tokens_counter[0] += n_tok

    final_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to("cpu")
            if torch.is_floating_point(v):
                v = v.to(torch.bfloat16)
        final_inputs[k] = v

    with torch.no_grad():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            model.model.language_model(**final_inputs)

    return n_tok


# ---------------------------------------------------------------------------
# Framework group weighting (used by run_accumulation_loop / run_profile)
# ---------------------------------------------------------------------------
# express and node records are merged into one 'backend' group so that the
# small node sub-corpus (6 records) is not a standalone group.
GROUP_REMAP = {
    "express": "backend",
    "node":    "backend",
}

# Target weights per group — sum to 1.0.
# Applied when a corpus has a 'framework' field; otherwise equal-weight
# averaging across all samples is used.
GROUP_WEIGHTS = {
    "html":    0.15,
    "css":     0.10,
    "react":   0.16,
    "vanilla": 0.15,
    "backend": 0.10,
    "vue":     0.10,
    "svelte":  0.08,
    "threejs": 0.08,
    "canvas":  0.08,
}


# ---------------------------------------------------------------------------
# Inner accumulation loop (shared by full-text and fw passes)
# ---------------------------------------------------------------------------
def run_accumulation_loop(samples, label, field_key="text", group_field=None):
    """
    Accumulate activations using per-sample normalisation so every sample
    contributes equally regardless of token length.

    Args:
        samples:     list of dicts
        label:       string for tqdm / checkpoint naming
        field_key:   dict key to read the text from ("text" or "framework")
        group_field: if set, samples[i][group_field] is mapped via GROUP_REMAP
                     to assign each sample to a named group for weighted
                     aggregation.  None → all samples assigned to "_all".

    Returns:
        (hist_prov, wsum_prov, n_tokens_total,
         group_freq_sums, group_wm_sums, group_counts)

        hist_prov       (L,E,K) int64   cumulative raw counts  (provenance)
        wsum_prov       (L,E,K) float32 cumulative raw wsum    (provenance)
        n_tokens_total  int             total tokens processed
        group_freq_sums dict[str -> (L,E,K) float32]  sum of per-sample freq
        group_wm_sums   dict[str -> (L,E,K) float32]  sum of per-sample wm
        group_counts    dict[str -> int]               samples per group
    """
    ckpt_hist = os.path.join(OUTPUT_DIR, f"{label}_hist_ckpt.pt")
    ckpt_wsum = os.path.join(OUTPUT_DIR, f"{label}_wsum_ckpt.pt")  # legacy; kept for cleanup
    ckpt_meta = os.path.join(OUTPUT_DIR, f"{label}_meta_ckpt.json")

    start_idx      = 0
    hist_provenance  = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.int64)
    wsum_provenance  = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
    hist_ew_prov     = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
    n_tokens_total   = 0
    group_freq_sums: dict = {}
    group_wm_sums:   dict = {}
    group_ew_sums:   dict = {}
    group_counts:    dict = {}

    if os.path.exists(ckpt_hist) and os.path.exists(ckpt_meta):
        try:
            state = torch.load(ckpt_hist, weights_only=False)
            hist_provenance.copy_(state["hist_prov"])
            wsum_provenance.copy_(state["wsum_prov"])
            if "hist_ew_prov" in state:
                hist_ew_prov.copy_(state["hist_ew_prov"])
            n_tokens_total  = state["n_tokens"]
            group_freq_sums = {k: v.clone() for k, v in state["group_freq_sums"].items()}
            group_wm_sums   = {k: v.clone() for k, v in state["group_wm_sums"].items()}
            group_ew_sums   = {k: v.clone() for k, v in state.get("group_ew_sums", {}).items()}
            group_counts    = dict(state["group_counts"])
            with open(ckpt_meta, "r") as f:
                meta = json.load(f)
            start_idx = meta.get("last_index", 0) + 1
            print(f"  [RESUME {label}] from index {start_idx}  "
                  f"groups={list(group_counts.keys())}")
        except Exception as e:
            print(f"  [WARN] checkpoint load failed ({e}), starting fresh")
            hist_provenance.zero_()
            wsum_provenance.zero_()
            n_tokens_total  = 0
            group_freq_sums = {}
            group_wm_sums   = {}
            group_counts    = {}
            start_idx = 0

    for i in tqdm(range(start_idx, len(samples)), desc=label):
        try:
            text = samples[i].get(field_key, "")
            if not text:
                continue

            # Reset per-sample hook accumulators before each forward pass
            current_histograms.zero_()
            current_weight_sum.zero_()
            current_hist_ew.zero_()
            n_tokens_counter.zero_()

            n_tok = max(run_forward_pass(text), 1)
            n_tokens_total += n_tok

            # Per-sample normalisation
            freq_sample = current_histograms.float() / n_tok
            wm_sample   = current_weight_sum / current_histograms.float().clamp(min=1)
            ew_sample   = current_hist_ew / n_tok    # avg p_sel contribution per token

            # Determine group
            if group_field is not None:
                raw_grp = samples[i].get(group_field, "_all") or "_all"
                grp = GROUP_REMAP.get(raw_grp, raw_grp)
            else:
                grp = "_all"

            if grp not in group_freq_sums:
                group_freq_sums[grp] = torch.zeros(
                    (NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
                group_wm_sums[grp]   = torch.zeros(
                    (NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
                group_ew_sums[grp]   = torch.zeros(
                    (NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
                group_counts[grp]    = 0

            group_freq_sums[grp] += freq_sample
            group_wm_sums[grp]   += wm_sample
            group_ew_sums[grp]   += ew_sample
            group_counts[grp]    += 1

            # Cumulative provenance (raw, for downstream calibration)
            hist_provenance += current_histograms
            wsum_provenance += current_weight_sum
            hist_ew_prov    += current_hist_ew

            if (i + 1) % args.checkpoint_every == 0:
                torch.save({
                    "hist_prov":       hist_provenance.clone(),
                    "wsum_prov":       wsum_provenance.clone(),
                    "hist_ew_prov":    hist_ew_prov.clone(),
                    "n_tokens":        n_tokens_total,
                    "group_freq_sums": {k: v.clone() for k, v in group_freq_sums.items()},
                    "group_wm_sums":   {k: v.clone() for k, v in group_wm_sums.items()},
                    "group_ew_sums":   {k: v.clone() for k, v in group_ew_sums.items()},
                    "group_counts":    dict(group_counts),
                }, ckpt_hist)
                with open(ckpt_meta, "w") as f:
                    json.dump({"last_index": i,
                               "n_tokens":   n_tokens_total,
                               "groups":     group_counts}, f)

        except Exception:
            import traceback
            print(f"  [Err] {label} sample {i}:")
            traceback.print_exc()
            continue

    for p in (ckpt_hist, ckpt_wsum, ckpt_meta):
        if os.path.exists(p):
            os.remove(p)

    return (hist_provenance, wsum_provenance, hist_ew_prov, n_tokens_total,
            group_freq_sums, group_wm_sums, group_ew_sums, group_counts)


# ---------------------------------------------------------------------------
# Profile runner — full pass + optional framework pass
# ---------------------------------------------------------------------------
def run_profile(data, domain_id):
    """
    Run full-text and (if applicable) framework-only accumulation for one domain.

    Returns a results dict with normalized freq/weight_mean plus raw tensors.
    """
    print(f"\n>>> Profile: {domain_id}  ({len(data)} samples)")

    # -- Full text pass ----------------------------------------------------
    hist_full, wsum_full, hist_ew_full, n_tokens_full, \
        group_freq_sums, group_wm_sums, group_ew_sums, group_counts = \
        run_accumulation_loop(
            data, label=f"{domain_id}__full", field_key="text", group_field="framework"
        )
    rank0_sum  = hist_full[:, :, 0].sum().item()
    total_sum  = hist_full.sum().item()
    print(f"  {domain_id}[full]: n_tokens={n_tokens_full:,}  "
          f"rank0_tokens={rank0_sum:,}  total_activations={total_sum:,}")

    # -- Framework-only pass (KB calibration corpora) ----------------------
    fw_data = [e for e in data if e.get("framework")]
    hist_fw  = torch.zeros_like(hist_full)
    wsum_fw  = torch.zeros_like(wsum_full)
    n_tokens_fw = 0

    if fw_data:
        print(f"  {domain_id}[fw]: {len(fw_data)} framework samples")
        hist_fw, wsum_fw, _, n_tokens_fw, _, _, _, _ = run_accumulation_loop(
            fw_data, label=f"{domain_id}__fw", field_key="framework", group_field=None
        )
        print(f"  {domain_id}[fw]: n_tokens={n_tokens_fw:,}")

    # -- Aggregate per-group means with optional weighting -----------------
    groups = set(group_counts.keys())
    if groups == {"_all"}:
        # No framework grouping — equal-weight average across all samples
        n = max(group_counts["_all"], 1)
        freq        = group_freq_sums["_all"] / n
        weight_mean = group_wm_sums["_all"]   / n
        ew_freq     = group_ew_sums["_all"]   / n
    else:
        # Weighted average of per-group means using GROUP_WEIGHTS
        freq        = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
        weight_mean = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
        ew_freq     = torch.zeros((NUM_LAYERS, NUM_EXPERTS, TOP_K), dtype=torch.float32)
        total_w = 0.0
        for grp, cnt in group_counts.items():
            w = GROUP_WEIGHTS.get(grp, 0.0)
            if w == 0.0 or cnt == 0:
                print(f"  [WARN] group '{grp}' weight={w} count={cnt} — skipping")
                continue
            freq        += (group_freq_sums[grp] / cnt) * w
            weight_mean += (group_wm_sums[grp]   / cnt) * w
            ew_freq     += (group_ew_sums[grp]   / cnt) * w
            total_w     += w
        if total_w < 1.0 - 1e-4:
            print(f"  [INFO] {domain_id}: total_w={total_w:.4f}, re-normalising")
        freq        /= max(total_w, 1e-9)
        weight_mean /= max(total_w, 1e-9)
        ew_freq     /= max(total_w, 1e-9)
        print(f"  {domain_id}[agg]: groups={sorted(group_counts.keys())}  "
              f"total_w={total_w:.4f}")

    return {
        "freq":        freq,          # (L, E, K) float32  — build_coe_mask input
        "weight_mean": weight_mean,   # (L, E, K) float32  — build_coe_mask input
        "ew_freq":     ew_freq,       # (L, E, K) float32  — entropy-weighted freq
        "histogram":   hist_full,     # (L, E, K) int64    — provenance
        "wsum":        wsum_full,     # (L, E, K) float32  — provenance
        "hist_ew":     hist_ew_full,  # (L, E, K) float32  — entropy-weighted provenance
        "n_tokens":    n_tokens_full,
        "n_samples":   len(data),
        "hist_fw":     hist_fw,       # (L, E, K) int64    — may be zeros
        "wsum_fw":     wsum_fw,       # (L, E, K) float32  — may be zeros
        "n_tokens_fw": n_tokens_fw,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = {}
if os.path.exists(OUTPUT_PT):
    try:
        results = torch.load(OUTPUT_PT, weights_only=False)
        completed = [k for k in results if not k.startswith("_")]
        print(f"\nLoaded existing results ({len(completed)} domains): {completed}")
    except Exception as e:
        print(f"[WARN] Could not load existing PT ({e}) — starting fresh")

# Write/update metadata
results["_meta"] = {
    "shape":       [NUM_LAYERS, NUM_EXPERTS, TOP_K],
    "num_layers":  NUM_LAYERS,
    "num_experts": NUM_EXPERTS,
    "top_k":       TOP_K,
    "model":       os.path.basename(args.model_path),
    "domains":     list(domain_registry.keys()),
}
torch.save(results, OUTPUT_PT)

status = load_status()

for domain_id, (path, text_spec) in active_domains.items():
    if status.get(domain_id) == "complete" and domain_id in results:
        print(f"\n[Skip] {domain_id} — already complete.")
        continue

    print(f"\n{'='*60}")
    print(f"Domain : {domain_id}")
    print(f"File   : {path}")
    print(f"Spec   : {text_spec}")
    print(f"{'='*60}")

    data = load_domain_data(path, text_spec)
    if not data:
        print(f"[ERROR] No data loaded for {domain_id} — skipping.")
        continue

    result = run_profile(data, domain_id)
    results[domain_id] = result
    torch.save(results, OUTPUT_PT)
    status[domain_id] = "complete"
    save_status(status)
    gc.collect()
    print_mem(f"POST-{domain_id}")
    print(f"[{domain_id}] saved -> {OUTPUT_PT}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Profiling complete")
print(f"{'='*60}")
print(f"Results file : {OUTPUT_PT}")
print(f"Status file  : {STATUS_FILE}")
print()
for domain_id in domain_registry:
    r = results.get(domain_id, {})
    n_s = r.get("n_samples", 0)
    n_t = r.get("n_tokens", 0)
    done = "✓" if status.get(domain_id) == "complete" else "✗"
    print(f"  [{done}] {domain_id:<35}  {n_s:>5} samples  {n_t:>10,} tokens")

print()
print("Next step:")
print(f"  python build_coe_mask.py --pt-file {OUTPUT_PT} --domain-weights <key:w,...> --output-name <label> --budget 64")

# Explicitly release the model so the OS reclaims ~52 GB before this process
# fully exits — prevents the next profiler subprocess from OOMing on launch.
print("\n[CLEANUP] Releasing model weights...")
try:
    for h in handles:
        h.remove()
except Exception:
    pass
del model
gc.collect()
print("[CLEANUP] Done.")
