"""Hash resources referenced by an existing first-query trace, without starting Kit.

Diagnostic only. This reads files and writes a JSON manifest. It does not import
Isaac Sim, initialize a GPU, render, alter caches, or change evaluation settings.
Runtime shader-cache contents and GPU shader/RNG state are outside this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(root: Path) -> dict:
    result = {"root": str(root), "exists": root.exists(), "files": {}, "errors": []}
    if not root.exists():
        return result
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        try:
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                first = handle.read(1024 * 1024)
                pointer = first.startswith(b"version https://git-lfs.github.com/spec/v1\n")
                block = first
                while block:
                    digest.update(block)
                    size += len(block)
                    block = handle.read(1024 * 1024)
            result["files"][relative] = {"size": size, "sha256": digest.hexdigest(), "lfs_pointer": pointer}
        except OSError as error:
            result["errors"].append({"path": relative, "error": str(error)})
    files = result["files"]
    result.update(file_count=len(files), total_bytes=sum(v["size"] for v in files.values()),
                  lfs_pointer_count=sum(v["lfs_pointer"] for v in files.values()),
                  tree_sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="Existing directory containing renderer_settings.json")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a fresh file name")
    settings = json.loads((args.capture / "renderer_settings.json").read_text())
    mdl_required = settings["/renderer"]["mdl"]["searchPaths"]["required"].split(";")
    if not mdl_required or not mdl_required[0]:
        parser.error("Trace does not specify an MDL resource search path")
    rtx = settings["/rtx"]
    groups = {
        "repo_assets": args.repo.resolve() / "assets",
        "kit_mdl": Path(mdl_required[0]).parent,
        "kit_rendering_data": Path(rtx["post"]["tonemap"]["ocio"]["cfgFilePath"]).parent.parent,
    }
    for index, value in enumerate(rtx["shaderDb"]["appShaderCachePathList"]):
        groups[f"packaged_shader_cache_{index}"] = Path(value).parent
    manifest = {
        "scope": "CPU-only resource content fingerprints from saved settings; no Kit/GPU/render/policy initialization",
        "not_observed": ["actual GPU shader RNG", "per-pass GPU frame counters", "runtime cache contents", "GPU intermediate buffers"],
        "groups": {},
    }
    for label, root in groups.items():
        print(f"Hashing {label}: {root}", flush=True)
        item = fingerprint(root)
        manifest["groups"][label] = item
        print(f"  {item.get('file_count', 0)} files, {item.get('total_bytes', 0)} bytes, {len(item['errors'])} errors", flush=True)
    manifest["complete"] = all(v["exists"] and not v["errors"] and not v["lfs_pointer_count"]
                               for v in manifest["groups"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.output}; complete={manifest['complete']}", flush=True)
    if not manifest["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
