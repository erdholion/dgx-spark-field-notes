# Running 1M-Context MoEs on 2× NVIDIA DGX Sparks (`GB10 sm_121a`): Co-Residency, Unified-Memory Guards & Real-Repo Coding Benchmarks

*Field notes, production scripts, and multi-file agentic coding benchmarks from a 2× NVIDIA DGX Spark (`GB10 sm_121a`, `aarch64`, `~121.7 GiB` unified RAM/node, `200 Gb/s` CX-7 RoCE) cluster — Updated October 31, 2026.*

**Provenance Tags:**
- `[measured-here]` — Measured directly on our 2× DGX Spark pair or workstation harness.
- `[third-party: X]` — Reported upstream by `X`; cited with exact issue/commit reference.
- `[arithmetic]` — Derived directly from shown inputs.

---

## 1. Live Production Topologies: 1-Spark Co-Resident vs. 2-Spark `TP=2`

Instead of dedicating both DGX Sparks to a single model 24/7, we run a **dual-topology catalog** behind a single OpenAI-compatible gateway (`:8088` → `:8888`):
- **Tier A — 2-Spark `TP=2` (`1M` / `200K+` Context):** Spans both nodes over `200 Gb/s` RoCE when maximum reasoning/coding depth or `1,048,576` context is needed.
- **Tier B — 1-Spark Co-Resident (`Node A` Only, `0 GiB` on `Node B`):** Runs `Qwen3.8-Flash-Next` (`NVFP4`) entirely on **Node A** (`76.5 GiB` used, `262,144` context × 3 slots) at **`71.8–76.6 tok/s` code**, leaving **Node B 100% free (`103+ GiB` available)** to co-host auxiliary GPU workloads (ComfyUI video/image generation, TTS/ASR microservices).

| Catalog Key | Model & Quantization | Topology | Engine & Patch Composition | Context / Slots | Measured Speed (`tok/s`) `[measured-here]` | Primary Role |
|---|---|---|---|---|---|---|
| **`ds41-1017-zig`** *(Daily Driver)* | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` + lossless `MXFP8` Engram (`190 GB`) | **2× Spark** (`TP=2`) | **Native Zig TensorFold** (`bertholomus/TensorFold@fbbc9713` + `hashd1ve` `0005–0013` + `MiaAI-Lab` `0014–0025`, `TF_DS_GRAPH_CAP=128`, `TF_DS_ENGRAM_LEAD=on`) | **1,048,576** shared pool × **4 slots** (`fp8` MLA + `q4_b32` K-tail) | **`93–103` code** / `~50` prose (`30s` load) | **#1 Daily Driver** (`10.0/10` on blind & zero-hint repo audits) |
| **`ds41-1017-uncensored-keys`** | Same base + `drowzeys/keys` 78-tensor `attn.wo_b` L10–35 overlay | **2× Spark** (`TP=2`) | Same Native Zig 1M engine + `<1s` reversible `TFDSRK01` rank-cache switcher ([`scripts/zig_keys_overlay.py`](scripts/zig_keys_overlay.py)) | **1,048,576** shared pool × **4 slots** | **`104.3` code** / `49.2` prose (`31s` load, `0%` refusal) | **Uncensored Daily Driver** (`0 GB` duplicate checkpoint storage) |
| **`qwen-flash`** | `nvidia/Qwen3.8-Flash-Next-FP4-Local` (`96 GiB`) | **1× Spark** (`Node A` only) | **Native Zig `qwen38-single-tf`** (`MiaAI-Lab` `tensorfold-1.1.4` + 19 patches + 20-core DMA loader) | **262,144** × **3 slots** (`BF16` KV, `76.5 GiB` Node A, **`0 GiB` Node B**) | **`71.8–76.6` code** / `~54` prose (`85s` load) | **1-Spark Co-Resident Mode** (leaves Node B 100% free) |
| **`qwen-flash-ablit`** | `Chompa1422/Qwen3.8-Flash-Next-NVFP4-Abliterated` (18/21 shards hardlinked) or `ablit_view.py` `house-qsa` sidecar | **1× Spark** (`Node A` only) | **Native Zig `qwen38-single-tf`** (`tensorfold-1.1.4`) | **262,144** × **3 slots** (`BF16` KV, **`0 GiB` Node B**) | **`74.4` code** / `39.8` prose (`85s` load, `0%` refusal) | **1-Spark Uncensored Co-Resident** (`~14 GB`–`190 MB` delta on disk) |
| **`glm-tf`** | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` | **2× Spark** (`TP=2`) | **MiaAI TensorFold v2** (`bf5f9452`, `DFlash2 + Copy`, `KV_POOL_GIB=8`, `expandable_segments:True`) | **200,000** (`1.01M` token KV pool) | **`58.9` c=1** / **`92.7` c=2** (`18s` load) | **Fast Multi-Edit Coding & High Concurrency** |
| **`glm-fly9i`** | `GLM-5.3-Flash-EXL3-TR3-4bpw` + `glm53-dflash2` | **2× Spark** (`TP=2`) | **`fly9i/spark-engine` v0.3.3** runtime refusal subspace ablation (`SPARK_ABLATE_MODE=16-40`, RoCE MTU 1024 patched) | **229,376–262,144** | **`40.1` decode** / `37.3` E2E (`0%` refusal) | **Runtime-Ablated Uncensored GLM** (zero static weight damage) |

---

## 2. Unified-Memory Survival: Priority Eviction Guard & Host RAM Reclamation

On `GB10` (`sm_121a`), **GPU memory is host RAM (`MemAvailable`)**, and CUDA driver allocations are not charged to any process's Linux RSS:
- **Why standard OOM protection fails:** When a unified-memory box runs out of `MemAvailable`, the Linux RSS-scoring OOM killer cannot see the 100 GiB CUDA allocation and kills small system/desktop daemons instead, wedging the node until a hard power cycle `[measured-here]`.
- **Why `--gpu-memory-utilization` is not a startup bound:** On `GB10`, vLLM `profile_run` can spike memory well above the requested utilization fraction (`vllm#46307`), and long prefills spike allocator headroom (`vllm#55569`).

### A. Two-Tier Priority Eviction Guard ([`scripts/memory_guard.py`](scripts/memory_guard.py))

Early single-threshold watchdogs (e.g., killing the LLM whenever `MemAvailable < 8 GiB`) caused false kills when a 1M-context model legitimately idled at `4.5–7.2 GiB` free or when a co-tenant app spiked on Node B `[measured-here]`. Our production [`scripts/memory_guard.py`](scripts/memory_guard.py) implements a **two-tier priority policy**:

1. **Pre-Launch & Soft-Pressure Eviction (`--evict-hard` or `MemAvailable < 8.0 GiB`):**
   - Before any 2-Spark `TP=2` model boots (`launch.py` calls `memory_guard.py --evict-hard` on both nodes), or if `MemAvailable` drops below `8.0 GiB` while a primary LLM is serving, the guard **gracefully unloads secondary GPU co-tenants first**:
     - Sends `/interrupt` + `/free` (`{"unload_models": true, "free_memory": true}`) to ComfyUI endpoints.
     - Sends `POST /unload` to auxiliary TTS/ASR/media services.
     - Terminates any lingering auxiliary GPU worker PIDs (`vllm-omni`, `hunyuan3d`, etc.) if `MemAvailable < 5.0 GiB`.
2. **Emergency Floor (`MemAvailable < 2.5–3.0 GiB` for 3 consecutive ticks after eviction):**
   - Only if `MemAvailable` remains below the `2.5–3.0 GiB` floor after evicting all auxiliary workloads does the guard kill the primary LLM container, writing a `guard-tripped` marker so systemd crash watchdogs stand down cleanly instead of entering a restart loop.

### B. Reclaiming `+3.16 GiB` of `MemAvailable` on Headless / Serving Nodes `[measured-here]`

When running `1,048,576` context across 4 slots on 2× Sparks, every GiB on Rank 0 (which also hosts the API gateway and rendezvous state) matters. On `2026-10-29`, auditing Node A while `ds41-1017-uncensored-keys` (`1M` ctx) was serving reclaimed **`+3.16 GiB` on Node A (`4.13 GiB` → `7.29 GiB` free)** and **`+1.19 GiB` on Node B (`8.12 GiB` → `9.31 GiB` free)**:

1. **Orphaned `/dev/shm` IPC segments (`+827 MiB`):** Previous crashed/stopped multi-process runs (`psm_*`, `sem.mp-*`, shared-memory mmap files) stayed in `/dev/shm`, directly subtracting from `MemAvailable`. Clean stale `/dev/shm` files in `stop.sh`.
2. **Bloated `dockerd` RSS (`1,224 MiB` → `108 MiB`, `+1.1 GiB`):** After repeated large image builds/pulls, `dockerd` held `1.22 GiB` RSS on Node A vs. `85 MiB` on Node B. Restarting `dockerd` during an idle maintenance window reclaimed `1.1 GiB`.
3. **Unused desktop/background daemons (`gdm`, `fwupd`, `packagekit`, `bluetooth`) + swap cycling:** Stopping headless desktop services and cycling swap (`swapoff -a && swapon -a` after `SwapCached` accumulated `652 MiB`) freed another `~1.2 GiB`.
4. **Capping `dsv41` CUDA Graph Cache (`TF_DS_GRAPH_CAP=128`, `MiaAI-Lab` patch `0033`):** Without a cap on `round.Graphs`, every unique prompt/decode shape captured over multi-day agentic sessions stays allocated in unified memory. Setting `TF_DS_GRAPH_CAP=128` (LRU eviction) bounds graph memory growth permanently.
5. **GPU Clock Lock (`300,2250 MHz`):** Locking `nvidia-smi -lgc 300,2250` via a boot systemd unit prevents `GB10` transient power-off spikes on `100K+` cold prefills while costing only `~1–2%` decode speed (`50–66 W`, `≤88 °C`) `[measured-here]`.

---

## 3. Real-Repository Multi-File Coding Benchmark Leaderboard (`[measured-here]`)

Synthetic single-file benchmarks (`HumanEval`) and greedy microbenchmarks (`sparkDash` with thinking off) do not reflect how local models behave inside real agentic coding harnesses (`OpenCode` / `OpenChamber` / `dsh`) across `85k–256k` token contexts.

We evaluated our local Spark models and engines across **3 real-repository multi-file engineering suites (22 full runs)** with automated SQLite telemetry extraction separating **pure decode `tok/s`**, **end-to-end (E2E) `tok/s`** (including prefill & cache reads), **speculative decoding acceptance (`tok/round`)**, **tool-call batching %**, **multi-edit turns**, and **shell anti-patterns (`sed`/`grep`/`tail` loops)**.

### Suite 1: Multi-File Feature & Bug-Fix Backlog Item (`~700` `pytest` + `10` Browser Smoke + `110` JS Regressions)

- **Prompt:** `"Take the next item on the backlog, fix it and test it."` (unscripted autonomous execution on a clean git clone).

#### Table 1A: Speed, Speculative Decoding & Wall-Clock Timing

| Rank & Model (`variant`) | Outcome | Wall Time | LLM / Tool Time | E2E `tok/s` (Gen Tok) | Pure Decode `tok/s` | Speculative Decoding (`tok/step`) |
|---|---|---|---|---|---|---|
| **1. `DeepSeek-V4.1-1017-Keys` (`#low`, Zig 1M)** | **PASS*** | **`11.37m`** | **`8.90m`** / `2.46m` | **`52.3`** (`27.9k`) | **`46.2`** (`ttft=3.51s`) | **G19 MTP: `2.62 tok/round`** (`mean_na=1.62`) |
| **2. `GLM-5.3-EXL3` (`#high`, `glm-tf` v2 `bf5f9452`)** | **PASS** | **`14.27m`** | **`10.16m`** / `4.11m` | **`39.8`** (`24.3k`) | **`~43.5`** (`v1=35ms`) | **DFlash2 + Copy: `~2.28 tok/step`** |
| **3. `DeepSeek-V4.1-Flash-TF` (`#high`, Run A)** | **PASS** | **`18.85m`** | `14.85m` / `4.00m` | **`48.5`** (`43.2k`) | **`49.6`** (`ttft=1.95s`) | **G19 MTP: `2.80 tok/round`** (`mean_na=1.80`) |
| **4. `DeepSeek-V4.1-Flash-TF` (`#low`, Clean)** | **PASS** | **`19.08m`** | **`7.86m`** / `11.22m` | **`42.2`** (`37.0k`) | **`52.4`** (`ttft=1.84s`) | **G19 MTP: `3.08 tok/round`** (`mean_na=2.08`) |
| **5. `GLM-5.3-EXL3` (`#low`, `glm-tf` v2 `bf5f9452`)** | **PASS** | **`20.21m`** | `11.24m` / `8.95m` | **`38.6`** (`26.0k`) | **`44.2`** (`ttft=0.98s`) | **DFlash2 + Copy: `2.97 tok/step`** (`61.3%` accept) |
| **6. `GLM-5.3-EXL3` (`#high`, `glm-tf` v1 `v1.10`)** | **PASS** | **`20.27m`** | `15.61m` / `4.66m` | `36.4` (`34.1k`) | `~40.5` | **DFlash2 + Copy: `~2.15 tok/step`** |
| **7. `Qwen3.8-Flash-Next` (`#medium`, 1-Spark Zig TF)** | **PASS** | **`25.61m`** | `18.16m` / `7.09m` | **`65.6`** (`71.4k`) | **`64.4`** (`ttft=1.23s`) | **TensorFold MTP-15: `3.20 tok/round`** (`62.2%` accept) |
| **8. `DeepSeek-V4.1-Flash-TF` (`#high`, Clean)** | **PASS** | **`29.30m`** | `19.31m` / `9.97m` | **`55.3`** (`81.0k`) | **`55.4`** (`ttft=2.50s`) | **G19 MTP: `3.10 tok/round`** (`mean_na=2.10`) |
| **9. `GLM-5.3-SparkGLM` (`#high`, `Atlas`)** | **PASS** | **`32.15m`** | `28.24m` / `3.91m` | `28.8` (`48.8k`) | `33.1` (`ttft=4.81s`) | **Atlas DFlash `k5`: `2.99 tok/step`** (`p1=73.2%`) |
| **10. `GLM-5.3-Uncensored-fly9i` (`#high`, `v0.3.3`)** | **PASS** | **`32.40m`** | `22.22m` / `10.17m` | **`37.3`** (`49.8k`) | **`40.1`** (`ttft=1.35s`) | **fly9i DFlash: `2.78 tok/step`** (`56.1%` accept) |
| **11. `DeepSeek-V4.1-1017-Keys` (`#high`, Clean)** | **PASS** | **`38.74m`** | `26.90m` / `11.80m` | **`48.9`** (`96.0k`) | **`49.1`** (`ttft=2.34s`) | **G19 MTP: `2.92 tok/round`** (`mean_na=1.92`) |
| **12. `DeepSeek-V4.1-Flash-TF` (`#high`, Extended)** | **PASS** | **`41.03m`** | `38.92m` / `2.11m` | `47.2` (`113.9k`) | `~48.5` | **G19 MTP: `3.17 tok/round`** (`mean_na=2.17`) |
| **13. `Qwen3.8-Flash-Next` (`#xhigh`, vLLM TP=2)** | **STOP** | `47.13m` | `44.78m` / `2.35m` | `35.1` (`94.4k`) | `31.4` (`ttft=0.92s`) | **vLLM MTP `k4`: `1.97 tok/step`** (`p1=49.2%`) |

#### Table 1B: Tool Discipline, Peak Context & Verification Scope

| Rank & Model (`variant`) | Turns | Batched Turns (`>=2` tools) | Multi-Edit Turns | Tool Calls (`read/grep/edit/sh`) | Shell Anti-Patterns (`sed/grep/tail/dbg`) | Peak Context | Scope & Test Verification |
|---|---|---|---|---|---|---|---|
| **1. `DeepSeek-V4.1-Keys` (`#low`)** | **`43`** | **`23.3%`** (`10/43`) | `0` | `6/0/16/29` | `20/13/12/0` | `110.9k` | `7` files (`+130/-13`), `702` pytest PASS |
| **2. `GLM-5.3-EXL3` (`#high` v2)** | **`59`** | **`27.1%`** (`16/59`) | **`4`** | `22/26/14/13` | **`1/3/0/0`** | **`85.9k`** | `4` files, `696` pytest, `10/10` smoke, `111` regr PASS |
| **3. `DeepSeek-V4.1` (`#high` A)** | `70` | `11.6%` (`8/69`) | `0` | `21/4/26/21` | `2/3/10/0` | `147.7k` | `6` files, `699` pytest, `110` regr PASS |
| **4. `DeepSeek-V4.1` (`#low` Clean)** | `93` | **`23.4%`** (`22/94`) | **`6`** | `35/23/28/29` | `4/1/0/2` | `155.2k` | `6` files, `699` pytest, `10/10` smoke, `110` regr PASS |
| **5. `GLM-5.3-EXL3` (`#low` v2)** | `96` | `14.6%` (`14/96`) | **`5`** | `32/25/25/26` | **`0/0/1/0`** | **`90.7k`** | Fixed 2 bugs (`6` files, `+125/-7`, `10/10` smoke, `110` regr) |
| **6. `GLM-5.3-EXL3` (`#high` v1)** | `91` | `10.1%` (`9/89`) | `3` | `37/31/14/18` | **`0/0/1/0`** | `116.2k` | `5` files, `696` pytest, `10/10` smoke, `110` regr PASS |
| **7. `Qwen3.8-Flash-Next` (`#medium` TF)** | `73` | **`38.4%`** (`28/73`) | **`10`** | `15/3/35/48` | `27/24/25/0` | `169.7k` | `10` files (`+203/-22`), `702` pytest, `10/10` smoke, `110` regr |
| **8. `DeepSeek-V4.1` (`#high` Clean)** | `112` | `17.9%` (`20/112`) | `3` | `29/19/48/36` | `9/6/9/0` | `221.6k` | `10` files (`+310/-40`), `705` pytest, `10/10` smoke, `109` regr |
| **9. `GLM-5.3-SparkGLM` (`#high`)** | `80` | **`64.6%`** (`51/79`) | `3` | `52/0/13/68` | `9/53/8/0` | `143.8k` | `4` files, `696` pytest, `10/10` smoke, `110` regr PASS |
| **10. `GLM-5.3-fly9i` (`#high` v0.3.3)** | `91` | **`24.2%`** (`22/91`) | **`4`** | `33/25/26/31` | `8/5/3/0` | `145.5k` | Fixed 2 bugs (`7` files, `+124/-15`, `10/10` smoke, `110` regr) |
| **11. `DeepSeek-V4.1-Keys` (`#high`)** | `167` | `10.2%` (`17/167`) | `0` | `31/0/43/108` | `50/45/46/0` | `238.1k` | **Most files (`13`):** `+300/-35`, all suites PASS |
| **12. `DeepSeek-V4.1` (`#high` Ext)** | `135` | `13.4%` (`18/134`) | `0` | `58/32/39/27` | `2/4/0/0` | `256.2k` | `8` files, `709` pytest, `10/10` smoke, `113` regr PASS |
| **13. `Qwen3.8-Flash-Next` (`#xhigh` vLLM)** | `65` | **`40.6%`** (`26/64`) | **`9`** | `31/16/24/16` | `0/3/4/0` | `165.9k` | Code PASS (`53/53` pytest); over-thought `76.7k` reasoning tok |

---

### Suite 2: 2-Phase Blind Repository Audit & 4-Step Implementation + `4×4` Cross-Model Unit-Test Matrix (`542`-Test Python + JS Codebase)

- **Design:**
  - **Phase 1 (Read-Only Blind Audit):** Each model audits a 542-test Python/JS web codebase cold (with all prior audit notes stripped), runs tests, and writes `BENCH-PLAN.md` identifying doc-vs-code discrepancies and must-fix bugs (scored against 10 canonical ground-truth bugs).
  - **Phase 2 (Canonical 4-Step Implementation in the Same Session):** Each model receives the identical 4-step multi-file specification (5 production Python/JS files + unit tests). Every model's resulting code is then tested in a **4×4 Cross-Validation Matrix** against the unit tests written by *all 4 models*.

| Model (`variant`) | Total Wall (`P1 + P2`) | Active LLM (`P1 + P2`) | E2E `tok/s` (`P1 / P2 / Tot`) | Speculative Decoding (`tok/round`) | Phase 1 Blind Audit Discovery (`/ 10`) | Phase 2 `4×4` Cross-Model Test Score (`/ 10`) |
|---|---|---|---|---|---|---|
| **`DeepSeek-V4.1-Flash-TF` (`#high`, Zig)** | `53.81m` (`30.7m + 23.1m`) | `37.07m` (`21.9m + 15.2m`) | `56.4` / **`69.9`** / `61.9` (`137.7k` tok) | **G19 MTP: `2.78`** | **`10.0 / 10` (Rank #1)** *(found all 10 bugs + live Playwright)* | **`9.0 / 10`** (`50/50` own tests PASS) |
| **`Qwen3.8-Flash-Next` (`#medium`, 1-Spark TF)** | `40.50m`* (`21.5m + 19.0m`) | **`20.57m`** (`12.9m + 7.7m`) | `56.1` / **`74.7`** / **`63.1`** (`77.8k` tok) | **TF MTP-15: `3.21`** (`63.6%` accept) | **`9.0 / 10` (Rank #2)** *(ran all 9 broken Playwright suites live)* | **`9.3 / 10` (Rank #2)** (`49/49` own tests PASS, preserved legacy fixtures) |
| **`DeepSeek-V4.1-1017-Keys` (`#low`, Zig)** | `39.57m` (`23.2m + 16.3m`) | `24.56m` (`13.4m + 11.2m`) | `54.9` / **`63.8`** / `58.9` (`86.8k` tok) | **G19 MTP: `2.48`** | **`8.5 / 10` (Rank #3)** *(ran all 14 `.mjs` suites live)* | **`10.0 / 10` (Rank #1)** *(passed 100% of all 4 models' edge-case tests)* |
| **`GLM-5.3-EXL3` (`#high`, `glm-tf` v2)** | **`27.78m`** (`13.3m + 14.5m`) | `20.59m` (`9.8m + 10.8m`) | `29.4` / `48.0` / `39.1` (`48.4k` tok) | **DFlash2 + Copy: `2.83`** | `4.5 / 10` (Rank #4) *(inspected statically; skipped live browser)* | `8.5 / 10` (`49/49` own tests PASS, `55.9%` batched turns) |

---

### Suite 3: Zero-Hint Status Audit & Natural Convergence (`542`-Test Codebase)

- **Prompt:** `"Analyze the current status of the project, write a concise BENCH-PLAN.md with your findings and prioritized next steps, and then implement and test the #1 most critical fix."`
- **Natural Convergence:** With zero hints, all 4 models independently ran the 542-test suite, identified the 1 failing test (`startswith("/tmp/")` path-escape bug when the repository checkout itself lived under `/tmp`, leaving untracked fixture files in the working tree), wrote a plan, and implemented a fix + regression test:
  - **`DeepSeek-V4.1-Flash-TF#high` (`10.0/10`, `13.40m` wall, `5.31m` LLM, `76.4 tok/s` E2E)** and **`Qwen3.8-Flash-Next#medium` (`10.0/10`, `12.66m` wall, `4.91m` LLM, `64.5 tok/s` E2E)** tied for **#1**: both discovered the live Playwright contract drift, fixed the `Path.is_relative_to(ROOT)` guard, added a dedicated unit test, and left `git status` 100% clean (`543/543` PASS).

---

## 4. Coding Harness Telemetry: `OpenCode`/`OpenChamber` vs. `DeepSeek Harness (dsh)` on `DeepSeek-V4.1-Flash-1017` (`[measured-here]`)

We ran sequential head-to-head sessions of `DeepSeek-V4.1-Flash-TF` (`#high`, Native Zig 1M) on the exact same 6-workstream, 23-file software engineering task using **`OpenChamber` (`OpenCode`)** vs. **`DeepSeek Harness` (`@deepseek-ai/dsh 0.2.1-rc.4`)**:

| Metric | `OpenChamber` (`OpenCode` + `AGENTS.md`) | `DeepSeek Harness` (`dsh headless`) | Delta / Finding |
|---|---|---|---|
| **Wall Time** | **`24.94 min`** (`1099s` LLM + `396s` tools) | `30.42 min` (`1825s`) | **`OpenChamber` `-5.48 min` (`18.0%` faster)** |
| **Assistant Turns** | **`144`** | `164` | **`-20` turns** |
| **Batched Turns (`>=2` tools/turn)** | **`31` (`21.5%`)** | `14` (`8.5%`) | **`2.5×` higher tool batching** |
| **Tool Calls (`read/grep/edit/bash`)** | `57 / 45 / 49 / 25` (`180` total) | `29 / 4 / 61 / 82` (`178` total) | Native `read`/`grep` vs. `82` `bash` calls |
| **Failed `edit` Calls** | **`0`** | `5` (`FS_NOT_OBSERVED`) | `bash` `sed`/`grep` reads don't register `fs/observed` in `dsh` |
| **Shell Anti-Patterns (`sed/grep/tail`)** | **`1 / 0 / 5`** | `30 / 31 / 39` | Global `AGENTS.md` rules eliminate shell read loops |
| **Generated Tokens & Peak Context** | **`83,581` tok** (`261.7K` peak ctx) | `103,092` tok (`280.5K` peak ctx) | **`-19.5K` generated tokens** |
| **E2E Throughput** | **`76.04 tok/s`** | `~75–77 tok/s` | Tie (same `ds41-1017-zig` 1M engine) |

### 5 Under-the-Hood Mechanisms from `@deepseek-ai/dsh` Worth Adopting in Any Local Harness

Inspecting `@deepseek-ai/dsh`'s internal packages (`dsh-llm-pi-ai`, `dsh-tool-fs`, `dsh-compaction-tool-result-pruner`, `dsh-repeat-tool-reminder`) revealed five post-training conventions that `DeepSeek-V4.1-1017` responds to directly:

1. **Wire-Level `chat_template_kwargs`:** Send `"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "high"}` (official DeepSeek 1017 levels in `encoding.py`: `low` = 50, `high` = 75, `max` = 100) rather than relying solely on top-level `reasoningEffort`.
2. **Zero-Echo `edit` and `write` Tool Results (`dsh-tool-fs`):** When an `edit` succeeds, return **only** `"The file <path> has been updated successfully."` to the model's conversation history, attaching full hunk diffs only to UI `presentationMeta`. Across 50–60 file edits, this saves tens of thousands of redundant diff tokens in the prompt prefix.
3. **DSML Streaming Parameter-Order Hints:** Because `DeepSeek-V4.1` uses live DSML tool streaming, include explicit parameter ordering in tool descriptions: *"Provide `file_path` before `old_string` and `new_string` in the arguments."*
4. **Automatic Tool-Result Pruner & Spill Policy:** Prune oversized tool outputs inline at `thresholdChars: 8192` (`headChars: 4096`, `tailChars: 1024`) and spill outputs exceeding `12,500` tokens to disk.
5. **Stateful Repeat-Tool-Call Loop Reminder:** Track repeated identical tool calls and inject escalating nudges at `thresholds: [3, 5, 8]`.

---

## 5. Included Utilities (`scripts/`)

- **[`scripts/memory_guard.py`](scripts/memory_guard.py)** — Two-tier unified-memory GPU priority & co-tenant eviction guard (`--list`, `--evict`, `--evict-hard`, and continuous daemon mode) for 1-Spark and 2-Spark deployments.
- **[`scripts/zig_keys_overlay.py`](scripts/zig_keys_overlay.py)** — Reversible `<1s` in-place `TFDSRK01` binary rank-cache overlay switcher for Native Zig `DeepSeek-V4.1-Flash-1017` (`stock <-> keys`, SHA-256 verified before and after write, `0 GB` duplicate rank cache).
- **[`scripts/patch_miner.py`](scripts/patch_miner.py)** — Compositional cross-fork `.patch` matrix scanner across DGX Spark community repositories (`bertholomus`, `hashd1ve`, `MiaAI-Lab`, `sehoon123`, `jayleaton`, `fly9i`).

## License

MIT
