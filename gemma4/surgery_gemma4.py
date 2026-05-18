"""
surgery_gemma4.py — Compact gemma4 Ollama GGUF to a domain-specialist K-expert subpool.

GGUF tensor layout (confirmed from inspection):
  ffn_gate_up_exps.weight : data shape (128, 1408, 1584) uint8  — axis 0 = expert
  ffn_down_exps.weight    : data shape (128, 2816,  748) uint8  — axis 0 = expert
  ffn_down_exps.scale     : data shape (128,)           f32    — axis 0 = expert
  ffn_gate_inp.weight     : data shape (128, 2816)       f32    — axis 0 = expert (router)
  ffn_gate_inp.scale      : data shape (2816,)           f32    — not per-expert, NOT sliced

Strategy: COMPACT REWRITE — write a new GGUF file with all non-selected expert rows
physically removed from stacked tensors.  Expert dim shrinks from 128 → K (e.g. 64) on
every per-expert tensor.  The metadata field gemma4.expert_count is updated to K.
llama.cpp allocates only K-expert weight buffers → ~(1 - K/128) × expert_weight
reduction in both file size and VRAM.

Detecting per-expert tensors: any blk.{l} tensor whose GGUF shape[-1] == n_experts
(e.g. 128) is treated as a stacked expert tensor and sliced on numpy axis 0.
This covers all four confirmed per-expert tensors without hardcoding.

Usage:
    python surgery_gemma4.py --mask mask_python_K64.json --src <blob_path> --out <out_path>
    python surgery_gemma4.py  # uses default paths
    python surgery_gemma4.py --dry-run  # verify layout + print savings estimate, no write

RAM note: sliced expert arrays (~8 GB for K=64 across 30 layers) are held in the
GGUFWriter tensor list before being flushed to disk.  Ensure ~10 GB free RAM.
"""

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType
import json
import numpy as np
import argparse
import re
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SRC  = Path(r"D:\ollama\models\blobs\sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df")
DEFAULT_MASK = Path(r"c:\RyzenAI\college of experts\gemma4\histograms\final\mask_python_K64.json")
DEFAULT_OUT  = Path(r"D:\ollama\models\blobs\gemma4-python-K64-q4_K_M.gguf")

N_EXPERTS_ORIG = 128

# ── Load mask ──────────────────────────────────────────────────────────────────

def load_mask(mask_path: Path) -> dict[int, list[int]]:
    """Returns {layer_idx: sorted list of kept expert IDs}."""
    with open(mask_path) as f:
        raw = json.load(f)
    return {int(k): sorted(v) for k, v in raw.items()}


# ── Generic KV field copy ──────────────────────────────────────────────────────

def _field_val(field):
    """Extract a Python value (or list for ARRAY) from a ReaderField."""
    ftype = field.types[0]
    if ftype == GGUFValueType.ARRAY:
        elem_type = field.types[1]
        parts, data = field.parts, field.data
        if elem_type == GGUFValueType.STRING:
            return [bytes(parts[i]).decode("utf-8") for i in data]
        elif elem_type in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
            return [float(parts[i][0]) for i in data]
        elif elem_type == GGUFValueType.BOOL:
            return [bool(parts[i][0]) for i in data]
        else:
            return [int(parts[i][0]) for i in data]
    elif ftype == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    elif ftype in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return float(field.parts[field.data[0]][0])
    elif ftype == GGUFValueType.BOOL:
        return bool(field.parts[field.data[0]][0])
    else:
        return int(field.parts[field.data[0]][0])


def copy_kv_fields(reader: GGUFReader, writer: GGUFWriter,
                   overrides: dict | None = None, skip: set | None = None):
    """
    Copy every KV field from reader to writer.
    overrides: {field_name: (new_val, vtype, sub_type|None)}
    skip:      field names to omit entirely (e.g. 'general.architecture' already
               written by GGUFWriter.__init__)
    """
    skip = skip or set()
    overrides = overrides or {}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        if key in overrides:
            new_val, vtype, sub_type = overrides[key]
            writer.add_key_value(key, new_val, vtype, sub_type)
            continue
        ftype = field.types[0]
        sub_type = field.types[1] if ftype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, _field_val(field), ftype, sub_type)


# ── Main surgery (compact rewrite) ────────────────────────────────────────────

def run_surgery(src: Path, out: Path, mask: dict[int, list[int]], dry_run: bool = False):
    K = len(next(iter(mask.values())))
    print(f"Source       : {src}")
    print(f"Output       : {out}")
    print(f"Mask         : {len(mask)} layers, K={K}  ({K}/{N_EXPERTS_ORIG} = {K/N_EXPERTS_ORIG:.0%} budget)")

    reader = GGUFReader(str(src), "r")
    arch = str(reader.fields["general.architecture"].parts[
        reader.fields["general.architecture"].data[0]
    ], "utf-8")
    print(f"Architecture : {arch}")
    print(f"Tensors in   : {len(reader.tensors)}")

    # Count bytes we'll save
    expert_bytes_orig = 0
    expert_bytes_new  = 0
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m and t.shape[-1] == N_EXPERTS_ORIG:
            layer = int(m.group(1))
            keep_k = len(mask.get(layer, []))
            nb = t.n_bytes
            expert_bytes_orig += nb
            expert_bytes_new  += nb * keep_k // N_EXPERTS_ORIG
    src_gb  = src.stat().st_size / 1024**3
    save_gb = (expert_bytes_orig - expert_bytes_new) / 1024**3
    est_out_gb = src_gb - save_gb
    print(f"Source size  : {src_gb:.2f} GB")
    print(f"Expert bytes : {expert_bytes_orig/1024**3:.2f} GB  ->  {expert_bytes_new/1024**3:.2f} GB  (save {save_gb:.2f} GB)")
    print(f"Estimated out: ~{est_out_gb:.2f} GB")

    if dry_run:
        print("\n[DRY RUN — no output written]")
        print("\nPer-layer expert tensor plan:")
        for l in sorted(mask.keys()):
            kept = mask[l]
            pruned = [e for e in range(N_EXPERTS_ORIG) if e not in set(kept)]
            print(f"  Layer {l:>2d}: keep {len(kept)} experts, prune {len(pruned)}")
        return

    print(f"\nWriting compact GGUF -> {out} ...")
    out.parent.mkdir(parents=True, exist_ok=True)

    # GGUFWriter.__init__ writes general.architecture automatically — skip it in copy
    writer = GGUFWriter(str(out), arch=arch, use_temp_file=False)

    copy_kv_fields(
        reader, writer,
        overrides={
            f"{arch}.expert_count": (K, GGUFValueType.UINT32, None),
        },
        skip={"general.architecture"},
    )

    # Add tensors: slice expert dims, copy everything else as-is
    n_expert_tensors = 0
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        is_expert = m is not None and t.shape[-1] == N_EXPERTS_ORIG

        if is_expert:
            layer = int(m.group(1))
            keep = mask.get(layer, list(range(N_EXPERTS_ORIG)))  # all if layer not in mask
            raw = t.data[keep]                         # (K, ...) byte buffer — axis 0 sliced, rest intact
            # No raw_shape: writer uses raw.shape (byte buffer) and derives logical shape via
            # quant_shape_from_byte_shape.  For float tensors (gate_inp, scale) shape IS logical.
            writer.add_tensor(t.name, raw, raw_dtype=t.tensor_type)
            n_expert_tensors += 1
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    print(f"  Expert tensors compacted : {n_expert_tensors}")
    print(f"  Non-expert tensors copied: {len(reader.tensors) - n_expert_tensors}")
    print("Flushing to disk (this may take a few minutes) ...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out_gb = out.stat().st_size / 1024**3
    print(f"\nOutput: {out}")
    print(f"Size  : {out_gb:.2f} GB  (was {src_gb:.2f} GB, saved {src_gb - out_gb:.2f} GB)")
    print("Done.")


# ── Verify: check that per-expert tensors are detectable ─────────────────────

def verify(src: Path, mask: dict[int, list[int]]):
    print(f"Verifying tensor layout in {src.name} ...")
    reader = GGUFReader(str(src), "r")

    layers = sorted(mask.keys())
    sample_layer = layers[len(layers) // 2]
    prefix = f"blk.{sample_layer}."

    found = [(t.name, t.shape.tolist(), t.data.shape)
             for t in reader.tensors
             if t.name.startswith(prefix) and t.shape[-1] == N_EXPERTS_ORIG]

    if not found:
        raise RuntimeError(f"No expert tensors found for layer {sample_layer} with shape[-1]=={N_EXPERTS_ORIG}")

    print(f"  Expert tensors detected for layer {sample_layer}:")
    for name, gguf_shape, data_shape in found:
        print(f"    {name}  GGUF shape={gguf_shape}  data.shape={list(data_shape)}")
        if data_shape[0] != N_EXPERTS_ORIG:
            raise RuntimeError(f"  FAIL: {name} axis 0 = {data_shape[0]}, expected {N_EXPERTS_ORIG}")

    K = len(mask[sample_layer])
    total_expert_tensors = sum(
        1 for t in reader.tensors
        if re.match(r"blk\.\d+\.", t.name) and t.shape[-1] == N_EXPERTS_ORIG
    )
    print(f"  Total expert tensors in file: {total_expert_tensors}")
    print(f"  Compaction: {N_EXPERTS_ORIG} -> {K} experts per layer ({K/N_EXPERTS_ORIG:.0%} budget)")
    print("Verification passed — surgery is safe to run.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--src",  type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify layout and print savings estimate without writing")
    args = parser.parse_args()

    mask = load_mask(args.mask)
    verify(args.src, mask)
    run_surgery(args.src, args.out, mask, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
