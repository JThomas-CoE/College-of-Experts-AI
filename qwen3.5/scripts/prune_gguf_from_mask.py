# NOTE (legacy): This file is intentionally unchanged.
# Expected mask format: list of 40 torch.LongTensor each shape (128,) — expert indices.
# If your mask is a BoolTensor (40, 256) from generate_python_mask.py, use the adapter:
#   scripts/build_python_specialist.py  (handles format conversion automatically)
"""
prune_gguf_from_mask.py  —  CoE surgical GGUF expert pruner, 256 → 128.

For each layer, only the 128 experts specified in the mask are retained.
Router gate weights are permuted to match the new expert ordering.
MTP head, shared experts, vision encoder, and all non-MoE tensors copy verbatim.

Usage (single build):
    python prune_gguf_from_mask.py \
        --mask   "masks/specialist/coverage_SQL_K128.pt" \
        --input  "D:/ollama/models/blobs/sha256-900dde62..." \
        --output-dir "D:/ollama/staged" \
        [--name CoE-SQL-35b-A3b-K128-q4_K_M]

After building, use --register to print Ollama registration instructions.
"""

import struct
import gc
import sys
import argparse
import numpy as np
from pathlib import Path

TARGET_EXPERTS = 128
GGUF_MAGIC     = 0x46554747   # 'GGUF' little-endian
CHUNK          = 64 * 1024 * 1024  # 64 MiB copy buffer


# ── GGUF I/O helpers ──────────────────────────────────────────────────────────

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def write_string(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def read_meta(f, vtype):
    if   vtype == 0:  return struct.unpack('<B',  f.read(1))[0]
    elif vtype == 1:  return struct.unpack('<b',  f.read(1))[0]
    elif vtype == 2:  return struct.unpack('<H',  f.read(2))[0]
    elif vtype == 3:  return struct.unpack('<h',  f.read(2))[0]
    elif vtype == 4:  return struct.unpack('<I',  f.read(4))[0]
    elif vtype == 5:  return struct.unpack('<i',  f.read(4))[0]
    elif vtype == 6:  return struct.unpack('<f',  f.read(4))[0]
    elif vtype == 7:  return struct.unpack('<?',  f.read(1))[0]
    elif vtype == 8:  return read_string(f)
    elif vtype == 9:
        at = struct.unpack('<I', f.read(4))[0]
        al = struct.unpack('<Q', f.read(8))[0]
        return (at, [read_meta(f, at) for _ in range(al)])
    elif vtype == 10: return struct.unpack('<Q',  f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q',  f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d',  f.read(8))[0]
    else: raise ValueError(f"Unknown GGUF vtype {vtype}")


def write_meta(f, vtype, val):
    if   vtype == 0:  f.write(struct.pack('<B',  val))
    elif vtype == 1:  f.write(struct.pack('<b',  val))
    elif vtype == 2:  f.write(struct.pack('<H',  val))
    elif vtype == 3:  f.write(struct.pack('<h',  val))
    elif vtype == 4:  f.write(struct.pack('<I',  val))
    elif vtype == 5:  f.write(struct.pack('<i',  val))
    elif vtype == 6:  f.write(struct.pack('<f',  val))
    elif vtype == 7:  f.write(struct.pack('<?',  val))
    elif vtype == 8:  write_string(f, val)
    elif vtype == 9:
        at, av = val
        f.write(struct.pack('<I', at))
        f.write(struct.pack('<Q', len(av)))
        for v in av: write_meta(f, at, v)
    elif vtype == 10: f.write(struct.pack('<Q',  val))
    elif vtype == 11: f.write(struct.pack('<q',  val))
    elif vtype == 12: f.write(struct.pack('<d',  val))


def blk_layer(name):
    """Return integer layer index for blk.N.* tensors, else None."""
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'blk' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


# ── Mask loading ──────────────────────────────────────────────────────────────

def load_catalog(mask_path):
    """
    Load a .pt mask file → dict {str(layer_idx): [128 int expert indices]}.
    File format: list of 40 torch.Tensor each shape (128,) dtype int64.
    """
    import torch
    masks = torch.load(mask_path, weights_only=False)
    assert isinstance(masks, list), f"Expected list, got {type(masks)}"
    catalog = {}
    for i, m in enumerate(masks):
        indices = m.tolist()
        if len(indices) != TARGET_EXPERTS:
            raise ValueError(
                f"Layer {i}: mask length {len(indices)} != {TARGET_EXPERTS}"
            )
        catalog[str(i)] = [int(x) for x in indices]
    print(f"Mask loaded:  {len(catalog)} layers × {TARGET_EXPERTS} experts")
    return catalog


# ── Surgery ───────────────────────────────────────────────────────────────────

def prune(input_path, output_path, catalog):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    file_size   = input_path.stat().st_size

    print(f"\n{'='*65}")
    print(f"Input  : {input_path.name}")
    print(f"        {file_size/1e9:.2f} GB")
    print(f"Output : {output_path}")
    print(f"{'='*65}\n")

    with open(input_path, 'rb') as fin:

        # ── Header ────────────────────────────────────────────────────────────
        magic     = struct.unpack('<I', fin.read(4))[0]
        version   = struct.unpack('<I', fin.read(4))[0]
        n_tensors = struct.unpack('<Q', fin.read(8))[0]
        n_meta    = struct.unpack('<Q', fin.read(8))[0]

        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#010x})")
        print(f"GGUF v{version}, {n_tensors} tensors, {n_meta} metadata keys")

        # ── Metadata ──────────────────────────────────────────────────────────
        meta_entries = []
        for _ in range(n_meta):
            key = read_string(fin)
            vt  = struct.unpack('<I', fin.read(4))[0]
            val = read_meta(fin, vt)
            meta_entries.append((key, vt, val))

        # ── Tensor infos ──────────────────────────────────────────────────────
        tensor_infos = []
        for _ in range(n_tensors):
            name   = read_string(fin)
            nd     = struct.unpack('<I', fin.read(4))[0]
            dims   = [struct.unpack('<Q', fin.read(8))[0] for _ in range(nd)]
            qtype  = struct.unpack('<I', fin.read(4))[0]
            offset = struct.unpack('<Q', fin.read(8))[0]
            tensor_infos.append({
                'name': name, 'dims': dims, 'qtype': qtype,
                'offset': offset, 'ndims': nd,
            })

        header_end = fin.tell()

        # ── Data start: probe alignment ────────────────────────────────────────
        data_start = None
        ALIGNMENT  = 32
        for align in [32, 64, 128, 256]:
            ds = ((header_end + align - 1) // align) * align
            fin.seek(ds)
            probe = fin.read(16)
            if len(probe) == 16 and any(b != 0 for b in probe):
                data_start = ds
                ALIGNMENT  = align
                break
        if data_start is None:
            data_start = ((header_end + 31) // 32) * 32
        print(f"Data start : offset {data_start}  (alignment {ALIGNMENT})")

        # ── Compute input byte sizes via sorted offsets ───────────────────────
        sorted_t = sorted(tensor_infos, key=lambda t: t['offset'])
        for i, t in enumerate(sorted_t):
            if i + 1 < len(sorted_t):
                t['size'] = sorted_t[i + 1]['offset'] - t['offset']
            else:
                t['size'] = file_size - (data_start + t['offset'])

        # ── Classify tensors & plan output layout ─────────────────────────────
        n_slice = n_route = n_copy = 0
        current_offset = 0
        output_tensors = []

        for t in tensor_infos:
            name     = t['name']
            is_mtp   = name.startswith('mtp.')
            is_shexp = 'shexp' in name
            t_out    = t.copy()

            if '_exps.' in name and not is_shexp and not is_mtp:
                # ── Stacked expert weight tensor ──────────────────────────────
                orig_experts = t['dims'][-1]
                assert orig_experts == 256, \
                    f"{name}: expected 256 experts on last dim, got {orig_experts}"
                slice_size = t['size'] // orig_experts
                assert slice_size * orig_experts == t['size'], \
                    f"{name}: size {t['size']} not divisible by {orig_experts}"
                t_out['action']     = 'slice_experts'
                t_out['slice_size'] = slice_size
                t_out['new_dims']   = list(t['dims'])
                t_out['new_dims'][-1] = TARGET_EXPERTS
                t_out['new_size']   = slice_size * TARGET_EXPERTS
                n_slice += 1

            elif 'gate_inp' in name and not is_shexp and not is_mtp:
                # ── MoE router weight: [hidden, num_experts] F32 ──────────────
                # In-memory layout is [Experts, Hidden] (PyTorch convention)
                orig_experts = t['dims'][-1]
                hidden       = t['dims'][0]
                assert orig_experts == 256, \
                    f"{name}: expected 256 on last dim, got {orig_experts}"
                t_out['action']   = 'permute_router'
                t_out['new_dims'] = [hidden, TARGET_EXPERTS]
                t_out['new_size'] = hidden * TARGET_EXPERTS * 4  # F32
                n_route += 1

            else:
                # ── Verbatim copy ─────────────────────────────────────────────
                # Covers: attention, norms, shexp, vision encoder, MTP (all),
                #   embeddings, output head, mtp.gate_inp (is_mtp guard above)
                t_out['action']   = 'copy'
                t_out['new_dims'] = t['dims']
                t_out['new_size'] = t['size']
                n_copy += 1

            # Align output offset
            t_out['new_offset'] = current_offset
            output_tensors.append(t_out)
            current_offset += t_out['new_size']
            current_offset = ((current_offset + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

        print(f"\nTensor plan:")
        print(f"  slice_experts  (blk._exps.)  : {n_slice:4d}")
        print(f"  permute_router (blk.gate_inp): {n_route:4d}")
        print(f"  copy verbatim  (all others)  : {n_copy:4d}")
        est_out = sum(t['new_size'] for t in output_tensors)
        print(f"\nEstimated output data: {est_out/1e9:.2f} GB  "
              f"(reduction {(1-est_out/file_size)*100:.1f}%)\n")

        # ── Write output GGUF ─────────────────────────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as fout:

            # Header
            fout.write(struct.pack('<I', magic))
            fout.write(struct.pack('<I', version))
            fout.write(struct.pack('<Q', n_tensors))
            fout.write(struct.pack('<Q', n_meta))

            # Metadata — patch expert_count (256 → 128), leave everything else
            patched = []
            for key, vt, val in meta_entries:
                write_string(fout, key)
                fout.write(struct.pack('<I', vt))
                if (key.endswith('expert_count')
                        and isinstance(val, int)
                        and val == 256):
                    write_meta(fout, vt, TARGET_EXPERTS)
                    patched.append(f"{key}: {val} → {TARGET_EXPERTS}")
                else:
                    write_meta(fout, vt, val)
            for p in patched:
                print(f"  Patched metadata: {p}")

            # Tensor infos (updated dims, offsets)
            for t in output_tensors:
                write_string(fout, t['name'])
                fout.write(struct.pack('<I', t['ndims']))
                for d in t['new_dims']:
                    fout.write(struct.pack('<Q', d))
                fout.write(struct.pack('<I', t['qtype']))
                fout.write(struct.pack('<Q', t['new_offset']))

            # Alignment pad after header section
            pos           = fout.tell()
            new_data_start = ((pos + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
            fout.write(b'\x00' * (new_data_start - pos))
            print(f"\nWriting tensors...")

            # Data blocks
            for i, t in enumerate(output_tensors):
                layer = blk_layer(t['name'])
                if layer is not None and str(layer) in catalog:
                    kept = catalog[str(layer)]
                else:
                    # MTP / non-blk tensors never reach slice/route actions
                    kept = list(range(TARGET_EXPERTS))

                fin.seek(data_start + t['offset'])

                # Align output position to match planned offset
                cur_pos = fout.tell() - new_data_start
                pad = t['new_offset'] - cur_pos
                if pad > 0:
                    fout.write(b'\x00' * pad)

                if t['action'] == 'copy':
                    rem = t['size']
                    while rem > 0:
                        chunk = fin.read(min(rem, CHUNK))
                        fout.write(chunk)
                        rem -= len(chunk)

                elif t['action'] == 'slice_experts':
                    full_data  = fin.read(t['size'])
                    slice_len  = t['slice_size']
                    out_buf    = bytearray(slice_len * TARGET_EXPERTS)
                    for new_i, old_i in enumerate(kept):
                        src = old_i * slice_len
                        dst = new_i * slice_len
                        out_buf[dst : dst + slice_len] = full_data[src : src + slice_len]
                    fout.write(out_buf)
                    del full_data, out_buf
                    gc.collect()

                elif t['action'] == 'permute_router':
                    full_data    = fin.read(t['size'])
                    orig_experts = t['dims'][-1]  # 256
                    hidden       = t['dims'][0]   # 2048
                    # Memory layout: [Experts, Hidden] (PyTorch row-major)
                    w     = np.frombuffer(full_data, dtype=np.float32).reshape(orig_experts, hidden)
                    new_w = np.empty((TARGET_EXPERTS, hidden), dtype=np.float32)
                    for new_i, old_i in enumerate(kept):
                        new_w[new_i] = w[old_i]
                    fout.write(new_w.tobytes())
                    del full_data, w, new_w
                    gc.collect()

                if (i + 1) % 200 == 0 or i == len(output_tensors) - 1:
                    pct = (i + 1) / len(output_tensors) * 100
                    print(f"  [{i+1:4d}/{len(output_tensors)}] {pct:5.1f}%"
                          f"  {t['name'][:60]}")

    out_size = output_path.stat().st_size
    print(f"\nOutput written: {out_size/1e9:.2f} GB "
          f"(reduction {(1 - out_size/file_size)*100:.1f}%)")
    return output_path


# ── Ollama registration helpers ───────────────────────────────────────────────

def write_modelfile(output_gguf, model_tag, modelfile_path, num_ctx=32768):
    """Write an Ollama Modelfile next to the staged GGUF."""
    content = (
        f"FROM {output_gguf}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        f"PARAMETER num_predict -1\n"
        f"PARAMETER num_gpu 99\n"
    )
    modelfile_path = Path(modelfile_path)
    modelfile_path.write_text(content, encoding='utf-8')
    print(f"\nModelfile written: {modelfile_path}")
    return modelfile_path


def print_registration_steps(output_gguf, model_tag, modelfile_path):
    print(f"\n{'='*65}")
    print("OLLAMA REGISTRATION")
    print(f"{'='*65}")
    print(f"\n  ollama create {model_tag}:latest -f \"{modelfile_path}\"")
    print(f"\nOllama will hash and store the GGUF in its blob store.")
    print(f"After successful registration you can delete the staged file:")
    print(f"  Remove-Item \"{output_gguf}\"")
    print(f"\nTest with:")
    print(f"  ollama run {model_tag}:latest")
    print(f"{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoE GGUF surgical pruner: 256 → 128 experts per layer"
    )
    parser.add_argument(
        '--mask', required=True,
        help="Path to coverage_*_K128.pt mask file"
    )
    parser.add_argument(
        '--input', required=True,
        help="Path to source GGUF or Ollama blob (qwen3.5-35b-a3b-q4_K_M)"
    )
    parser.add_argument(
        '--output-dir', required=True,
        help="Directory for staged output GGUF (e.g. D:/ollama/staged)"
    )
    parser.add_argument(
        '--name', default=None,
        help="Output filename stem (auto-derived from mask if omitted)"
    )
    parser.add_argument(
        '--num-ctx', type=int, default=32768,
        help="Context length for Modelfile (default: 32768)"
    )
    args = parser.parse_args()

    mask_path = Path(args.mask)

    # Auto-derive output name from mask filename
    if args.name:
        stem = args.name
    else:
        # coverage_SQL_K128.pt  →  CoE-SQL-35b-A3b-K128-q4_K_M
        stem = (mask_path.stem
                .replace('coverage_', 'CoE-')
                .replace('_K128', '-35b-A3b-K128-q4_K_M'))

    output_dir  = Path(args.output_dir)
    output_gguf = output_dir / f"{stem}.gguf"
    model_tag   = stem.lower()
    modelfile   = output_dir / f"Modelfile-{stem}.txt"

    catalog = load_catalog(mask_path)
    prune(args.input, output_gguf, catalog)

    write_modelfile(output_gguf, model_tag, modelfile, num_ctx=args.num_ctx)
    print_registration_steps(output_gguf, model_tag, modelfile)


if __name__ == '__main__':
    main()
