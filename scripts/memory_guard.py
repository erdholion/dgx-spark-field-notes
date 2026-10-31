#!/usr/bin/env python3
"""
memory_guard.py — Priority-aware Unified-Memory GPU Guard for NVIDIA DGX Spark (GB10 sm_121a).

Design:
  1. Primary LLM serving containers/processes (DeepSeek-V4.1, Qwen3.8-Flash-Next, GLM-5.3-Flash)
     are PROTECTED down to an emergency floor (default: 2.5 GiB MemAvailable).
  2. Auxiliary co-tenant GPU workloads (e.g., ComfyUI, TTS/ASR servers, secondary workers on Node B)
     are EVICTABLE:
       - Before launching a 2-Spark TP=2 model, call `memory_guard.py --evict-hard`.
       - During continuous daemon monitoring, if MemAvailable drops below the soft threshold
         (default: 8.0 GiB), co-tenant HTTP unload endpoints (/free, /unload) are called first.
       - Only if MemAvailable stays below the emergency floor (default: 2.5 GiB) for 3 consecutive
         ticks after evicting co-tenants does the guard stop the primary LLM and write a
         `guard-tripped` marker so crash watchdogs stand down cleanly.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SOFT_EVICT_GIB = float(os.environ.get("SPARK_GUARD_SOFT_GIB", "8.0"))
HARD_EVICT_GIB = float(os.environ.get("SPARK_GUARD_HARD_GIB", "5.0"))
EMERGENCY_FLOOR_GIB = float(os.environ.get("SPARK_GUARD_FLOOR_GIB", "2.5"))

PROTECTED_DOCKER_FILTERS = [
    "label=local-ai-kit=spark-model-manager",
    "label=tf.patches",
    "name=^dsv41-tf-",
    "name=^dsv41-1017-tf$",
    "name=^glm53-flash-tf$",
    "name=^qwen38-fn-tf$",
    "name=^qwen38-flash-next-tf$",
]

EVICTABLE_PROCESS_PATTERNS = (
    "vllm-omni",
    "f5_server.py",
    "hunyuan3d",
    "main.py --reserve-vram",
)

EVICTABLE_CONTAINERS = (
    "comfyui-worker",
)

COMFYUI_PORTS = tuple(
    int(p) for p in os.environ.get("SPARK_GUARD_COMFY_PORTS", "8188,8190").split(",") if p.strip()
)
UNLOAD_URLS = tuple(
    u.strip() for u in os.environ.get("SPARK_GUARD_UNLOAD_URLS", "http://127.0.0.1:8078/api/gpu/free,http://127.0.0.1:8765/unload").split(",") if u.strip()
)


def mem_available_bytes() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def protected_targets() -> dict:
    found = {}
    for flt in PROTECTED_DOCKER_FILTERS:
        r = subprocess.run(
            ["docker", "ps", "--filter", flt, "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if line.strip():
                cid, name = line.strip().split(" ", 1)
                found[cid] = name
    r2 = subprocess.run(
        ["pgrep", "-f", "[s]park-engine serve"],
        capture_output=True, text=True, timeout=5,
    )
    for pid in r2.stdout.splitlines():
        if pid.strip():
            found["pid:" + pid.strip()] = "spark-engine"
    return found


def _http_post(url: str, payload=None, timeout: float = 2.5):
    try:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except Exception:
        return None


def evictable_gpu_pids() -> dict:
    pids = {}
    self_pids = {os.getpid(), os.getppid()}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if pid in self_pids:
                continue
            try:
                cmdline = (
                    pathlib.Path(f"/proc/{pid}/cmdline")
                    .read_bytes()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", "ignore")
                )
            except OSError:
                continue
            if "memory_guard.py" in cmdline or "memory-guard.py" in cmdline:
                continue
            if any(pat in cmdline for pat in EVICTABLE_PROCESS_PATTERNS):
                pids[pid] = cmdline.strip()[:120]
    except Exception:
        pass
    return pids


def evict_cotenants(hard: bool = False) -> list:
    actions = []
    for port in COMFYUI_PORTS:
        if _http_post(f"http://127.0.0.1:{port}/interrupt") is not None:
            actions.append(f"comfyui:{port}:interrupt")
        if _http_post(f"http://127.0.0.1:{port}/free", {"unload_models": True, "free_memory": True}) is not None:
            actions.append(f"comfyui:{port}:free")
    for url in UNLOAD_URLS:
        if _http_post(url) is not None:
            actions.append(f"unload:{url}")
    if actions:
        time.sleep(1.5)
    for pid, desc in list(evictable_gpu_pids().items()):
        if hard:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=5)
            actions.append(f"kill:{pid}:{desc[:40]}")
    if hard and mem_available_bytes() < int(105 * 1024**3):
        for cname in EVICTABLE_CONTAINERS:
            r = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"name=^{cname}$"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                subprocess.run(["docker", "stop", "-t", "2", cname], capture_output=True, timeout=10)
                actions.append(f"docker-stop:{cname}")
    return actions


def main():
    if sys.argv[1:] == ["--list"]:
        print(json.dumps(protected_targets()))
        return
    if sys.argv[1:] in (["--evict"], ["--evict-hard"]):
        hard = sys.argv[1] == "--evict-hard"
        acts = evict_cotenants(hard=hard)
        print(json.dumps({"evicted": acts, "available_gib": round(mem_available_bytes() / 1024**3, 2)}))
        return

    low_ticks = 0
    floor_bytes = int(EMERGENCY_FLOOR_GIB * 1024**3)
    soft_bytes = int(SOFT_EVICT_GIB * 1024**3)
    hard_bytes = int(HARD_EVICT_GIB * 1024**3)

    while True:
        available = mem_available_bytes()
        active_full = protected_targets()
        if active_full and available < soft_bytes:
            evicted = evict_cotenants(hard=(available < hard_bytes))
            if evicted:
                print(
                    json.dumps({
                        "time": time.time(),
                        "protected": active_full,
                        "evicted": evicted,
                        "available_bytes": available,
                    }),
                    flush=True,
                )
                available = mem_available_bytes()

        low_ticks = low_ticks + 1 if available < floor_bytes else 0
        if low_ticks >= 3:
            evict_cotenants(hard=True)
            time.sleep(1.0)
            available = mem_available_bytes()
            if available < floor_bytes and active_full:
                event = {"time": time.time(), "available_bytes": available, "stopped": active_full}
                (HERE / "guard-tripped").write_text(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                cids = [k for k in active_full if not k.startswith("pid:")]
                if cids:
                    subprocess.run(["docker", "kill", *cids], timeout=20)
                if any(k.startswith("pid:") for k in active_full):
                    subprocess.run(["pkill", "-9", "-f", "[s]park-engine serve"], timeout=10)
            low_ticks = 0
        time.sleep(3)


if __name__ == "__main__":
    main()
