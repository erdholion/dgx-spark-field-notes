#!/usr/bin/env python3
"""
zig_keys_overlay.py — Reversible <1s L10-35 attn.wo_b Keys overlay switcher on a
Native Zig TensorFold TP=2 rank cache (TFDSRK01 format, rank 0 and rank 1).

Swaps the 78 EXL3 tensor slices (layers 10..35 attn.wo_b.{tr,suh,svh}) in-place
inside `tf-ds-rank/rank{0,1}of2-*.bin` with SHA-256 verification before and after,
avoiding a duplicate ~200 GB rank cache across the two nodes.
"""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_write(p: Path, b: bytes):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


class Sidecar:
    def __init__(self, path: Path):
        self.f = path.open("rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def get(self, key: str) -> bytes:
        lo, hi = self.header[key]["data_offsets"]
        self.f.seek(self.base + lo)
        return self.f.read(hi - lo)


def rank_slice(sc: Sidecar, layer: int, part: str, rank: int) -> bytes:
    sp_part = "trellis" if part == "tr" else part
    raw = sc.get(f"layers.{layer}.attn.wo_b.{sp_part}")
    if part == "tr":
        half = 256 * 320 * 80 * 2
        return raw[:half] if rank == 0 else raw[half:]
    if part == "suh":
        half = 4096 * 2
        return raw[:half] if rank == 0 else raw[half:]
    if part == "svh":
        return raw[:5120 * 2]
    raise ValueError(part)


def find_rank_bin(cache_dir: Path, rank: int) -> Path:
    matches = list((cache_dir / "tf-ds-rank").glob(f"rank{rank}of2-*.bin"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected 1 rank{rank}of2-*.bin in {cache_dir}/tf-ds-rank, found {matches}")
    return matches[0]


def read_index(bin_path: Path) -> dict:
    with bin_path.open("rb") as f:
        f.seek(-16, 2)
        end = f.tell()
        at, magic = struct.unpack("<Q8s", f.read(16))
        assert magic == b"TFDSRK01", f"Bad magic in {bin_path}"
        f.seek(at)
        rows = json.loads(f.read(end - at))
    return {r[0]: {"dtype": r[1], "shape": r[2], "offset": r[3], "nbytes": r[4]} for r in rows}


def main():
    ap = argparse.ArgumentParser(description="Reversible <1s Keys overlay switcher for Native Zig TFDSRK01 rank cache.")
    ap.add_argument("action", choices=("prepare", "switch", "status"))
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    ap.add_argument("--mode", choices=("stock", "keys"), default="stock")
    ap.add_argument("--cache", type=Path, required=True, help="Path to directory containing tf-ds-rank/")
    ap.add_argument("--state", type=Path, required=True, help="Path to store extracted stock/keys slices")
    ap.add_argument("--sidecar", type=Path, required=True, help="Path to exl3_wo_b_l10_35_mia29.safetensors")
    ap.add_argument("--model", type=Path, required=True, help="Path to base EXL3 model directory")
    args = ap.parse_args()

    bin_path = find_rank_bin(args.cache, args.rank)
    sdir = args.state / f"rank{args.rank}"
    sdir.mkdir(parents=True, exist_ok=True)
    meta_p = sdir / "overlay_meta.json"

    if args.action == "prepare":
        if meta_p.exists():
            print(f"rank{args.rank}: zig overlay already prepared at {sdir}")
            return
        idx = read_index(bin_path)
        sc = Sidecar(args.sidecar)
        wmap = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
        col_slice = "(0, 4096)" if args.rank == 0 else "(4096, 8192)"
        entries = []
        with bin_path.open("rb") as f_bin:
            for layer in range(10, 36):
                orig_sc = Sidecar(args.model / wmap[f"layers.{layer}.attn.wo_b.trellis"])
                for part in ("tr", "suh", "svh"):
                    key = f"layers.{layer}.attn.wo_b|None|{col_slice}|{part}"
                    st = idx[key]
                    off, nbytes = st["offset"], st["nbytes"]
                    f_bin.seek(off)
                    cur = f_bin.read(nbytes)
                    exp_stock = rank_slice(orig_sc, layer, part, args.rank)
                    assert cur == exp_stock, f"Stock mismatch at layer {layer} {part}"
                    new_keys = rank_slice(sc, layer, part, args.rank)
                    assert len(new_keys) == nbytes
                    eid = f"L{layer}_{part}"
                    atomic_write(sdir / f"{eid}.stock", cur)
                    atomic_write(sdir / f"{eid}.keys", new_keys)
                    entries.append({
                        "id": eid,
                        "offset": off,
                        "nbytes": nbytes,
                        "stock_sha": sha256(cur),
                        "keys_sha": sha256(new_keys),
                    })
                orig_sc.f.close()
        atomic_write(meta_p, json.dumps({"bin_path": str(bin_path), "entries": entries}, indent=2).encode())
        atomic_write(sdir / "active_mode", b"stock")
        print(f"rank{args.rank}: prepared {len(entries)} entries for {bin_path.name}.")
        return

    meta = json.loads(meta_p.read_text())
    if args.action == "status":
        mode = (sdir / "active_mode").read_text().strip() if (sdir / "active_mode").exists() else "unknown"
        print(f"rank{args.rank}: active_mode={mode}")
        return

    if args.action == "switch":
        target = args.mode
        with bin_path.open("r+b") as f_bin:
            for e in meta["entries"]:
                blob = (sdir / f"{e['id']}.{target}").read_bytes()
                assert sha256(blob) == e[f"{target}_sha"]
                f_bin.seek(e["offset"])
                f_bin.write(blob)
            f_bin.flush()
            os.fsync(f_bin.fileno())
            for e in meta["entries"]:
                f_bin.seek(e["offset"])
                assert sha256(f_bin.read(e["nbytes"])) == e[f"{target}_sha"]
        atomic_write(sdir / "active_mode", target.encode())
        print(f"rank{args.rank}: switched {bin_path.name} to {target} (78 tensors verified).")


if __name__ == "__main__":
    main()
