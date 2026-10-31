#!/usr/bin/env python3
"""
patch_miner.py — Compositional Recipe & Cross-Fork Patch Miner for DGX Spark (GB10 sm_121a).

Scans `patches/`, `recipe/patches/`, and `docker/patches/` directories, branches, and recent
commits across community forks for DeepSeek-V4.1-Flash-1017, Qwen3.8-Flash-Next, and
GLM-5.3-Flash, outputting `G_patch_matrix.json` and `G_patch_matrix.md`.
"""
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

FAMILIES = {
    "DS4.1-1017 (2x Spark Native Zig & Overlays)": {
        "deployed_recipe": "bertholomus/TensorFold@fbbc9713 + hashd1ve 0005..0013 + MiaAI 0014..0025 + drowzeys/keys",
        "deployed_patches": [
            "0005..0013 (hashd1ve)",
            "0014-c7-indexer-split",
            "0015-rdma-idle-nap",
            "0016..0019 (keepalive + graph-cap 128)",
            "0020..0025 (parked-pages, engram-lead, metrics, rendezvous, aligned-index)",
        ],
        "repos": [
            "hashd1ve/deepseek-v4.1-flash-2x-dgx-spark",
            "MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks",
            "MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold",
            "sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned",
            "bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10",
            "bertholomus/TensorFold",
            "jayleaton/deepseek-v41-tensorfold-spark",
            "harshilmathur/sparklab",
        ],
    },
    "Qwen3.8-Flash-Next (1-Spark Co-Resident & 2-Spark TP2)": {
        "deployed_recipe": "MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold (tensorfold-1.1.4 + ablit_view.py)",
        "deployed_patches": [
            "0001..0019 (tensorfold-1.1.4 single-spark)",
        ],
        "repos": [
            "MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold",
            "MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold",
            "BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks",
            "XtraSaltyDev/qwen38-flash-next-ablit",
            "ashhart/TensorFold",
        ],
    },
    "GLM-5.3-Flash (2x Spark TensorFold & Runtime Ablation)": {
        "deployed_recipe": "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ bf5f9452 & fly9i/spark-engine",
        "deployed_patches": ["0001..0131 (bf5f9452)", "fly9i-ibv-mtu-1024"],
        "repos": [
            "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold",
            "fly9i/spark-engine",
            "Enntity/sparkglm",
            "kindlingai/glm-5.3-flash-exlr8-gx10",
            "myllmbox/glm53-flash-cluster-recipe",
        ],
    },
}


def gh_get(path: str):
    url = f"https://api.github.com/{path.lstrip('/')}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "dgx-spark-patch-miner"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def inspect_repo(repo: str) -> dict:
    info = {"repo": repo, "branches": [], "patches": [], "recent_commits": []}
    try:
        branches = gh_get(f"repos/{repo}/branches?per_page=30")
        info["branches"] = [{"name": b["name"], "sha": b["commit"]["sha"][:12]} for b in branches]
    except Exception as e:
        info["branches_error"] = str(e)

    for pdir in ["patches", "recipe/patches", "docker/patches"]:
        try:
            items = gh_get(f"repos/{repo}/contents/{pdir}")
            if isinstance(items, list):
                info["patches"].extend(
                    {"path": x["path"], "name": x["name"], "sha": x["sha"][:12], "size": x.get("size", 0)}
                    for x in items
                    if x.get("name", "").endswith((".patch", ".diff"))
                )
        except Exception:
            pass

    try:
        commits = gh_get(f"repos/{repo}/commits?per_page=8")
        if isinstance(commits, list):
            info["recent_commits"] = [
                {
                    "sha": c["sha"][:12],
                    "date": c["commit"]["committer"]["date"],
                    "title": c["commit"]["message"].splitlines()[0][:140],
                }
                for c in commits
            ]
    except Exception as e:
        info["commits_error"] = str(e)
    return info


def main():
    matrix = {}
    md_lines = ["# Cross-Fork Patch & Compositional Recipe Matrix", ""]
    for fam, cfg in FAMILIES.items():
        md_lines.append(f"## {fam}")
        md_lines.append(f"- **Deployed Baseline:** `{cfg['deployed_recipe']}`")
        md_lines.append(f"- **Deployed Patches:** `{', '.join(cfg['deployed_patches'])}`")
        md_lines.append("")
        fam_out = []
        for repo in cfg["repos"]:
            rinfo = inspect_repo(repo)
            fam_out.append(rinfo)
            b_str = ", ".join(f"`{b['name']}@{b['sha']}`" for b in rinfo.get("branches", [])[:5]) or "n/a"
            p_names = [p["name"] for p in rinfo.get("patches", [])]
            md_lines.append(f"### `{repo}`")
            md_lines.append(f"- **Branches:** {b_str}")
            if p_names:
                md_lines.append(f"- **Patches ({len(p_names)}):** `{', '.join(p_names[-12:])}`")
            for c in rinfo.get("recent_commits", [])[:4]:
                md_lines.append(f"  - `{c['sha']}` ({c['date'][:10]}): {c['title']}")
            md_lines.append("")
        matrix[fam] = fam_out

    (HERE / "G_patch_matrix.json").write_text(json.dumps(matrix, indent=2))
    (HERE / "G_patch_matrix.md").write_text("\n".join(md_lines))
    print("Wrote G_patch_matrix.json and G_patch_matrix.md")


if __name__ == "__main__":
    main()
