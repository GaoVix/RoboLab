"""Offline, single-call OpenPI diagnostic; never changes the evaluation runner.

Use the same deterministic integer-image fixture on both machines to isolate the
policy from rendering. This fixture is not an observation from an evaluation.
An optional NPZ request can instead supply recorded RGB/state/prompt arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from reproducible_settings import POLICY_CONFIG, ROOT, policy_environment, profile_fingerprint


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fixture(np):
    y, x = np.indices((224, 224), dtype=np.uint16)
    exterior = np.stack(((x + 2*y) % 256, (3*x + y) % 256, (5*x + 7*y) % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(((7*x + y + 19) % 256, (x + 3*y + 31) % 256,
                      (2*x + 5*y + 47) % 256), axis=-1).astype(np.uint8)
    return {"observation/exterior_image_1_left": exterior,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": np.array([0, -0.63, 0, -2.51, 0, 1.87, 0], dtype=np.float32),
            "observation/gripper_position": np.array([0], dtype=np.float32),
            "prompt": "Put all plastic bottles away in the bin"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=ROOT.parent / "openpi-robolab")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / ".cache/openpi/openpi-assets-simeval/pi05_droid_jointpos")
    parser.add_argument("--request", type=Path, help="NPZ with observation__* keys and a prompt string")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-hlo", action="store_true", help="Save lowered and optimized HLO after the measured call")
    args = parser.parse_args()
    if not 0 <= args.policy_seed < 2**32:
        parser.error("policy seed must be in [0, 2**32)")
    openpi = args.openpi_root.resolve()
    environment = policy_environment(openpi)
    for name in list(os.environ):
        if name not in environment:
            os.environ.pop(name)
    os.environ.update(environment)
    sys.path.insert(0, str(openpi / "src"))

    import jax
    import numpy as np

    def digest(value):
        if isinstance(value, str):
            return {"text": value, "sha256": hashlib.sha256(value.encode()).hexdigest()}
        array = np.asarray(value)
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    print("Diagnostic only: one original policy.infer(request), fresh noise from the seeded RNG; no simulator.", flush=True)
    print("Hashing checkpoint files...", flush=True)
    weights = {str(p.relative_to(checkpoint)): file_hash(p)
               for p in sorted((checkpoint / "params").rglob("*")) if p.is_file()}
    if not weights:
        raise RuntimeError("No checkpoint parameter files found")
    for p in sorted((checkpoint / "assets").rglob("norm_stats.json")):
        weights[str(p.relative_to(checkpoint))] = file_hash(p)
    sources = {str(p.relative_to(openpi)): file_hash(p)
               for p in sorted((openpi / "src/openpi").rglob("*.py"))}
    sources["scripts/serve_policy.py"] = file_hash(openpi / "scripts/serve_policy.py")
    packages = {}
    for name in ["jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "numpy",
                 "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvcc-cu12",
                 "nvidia-cuda-runtime-cu12", "pillow", "opencv-python", "sentencepiece"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    request = fixture(np)
    if args.request is not None:
        with np.load(args.request, allow_pickle=False) as packed:
            request = {name: packed[name.replace("/", "__")].copy() for name in request if name != "prompt"}
            request["prompt"] = str(packed["prompt"].item())
    np.savez_compressed(output / "request.npz", **{k.replace("/", "__"): v for k, v in request.items()})

    spec = importlib.util.spec_from_file_location("official_serve_policy_diagnostic", openpi / "scripts/serve_policy.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    policy = official.create_policy(official.Args(policy=official.Checkpoint(
        config=POLICY_CONFIG, dir=str(checkpoint))))
    if policy._is_pytorch_model or policy._sample_kwargs:
        raise RuntimeError("Expected the unmodified JAX policy with default sampling arguments")
    policy._rng = jax.random.key(args.policy_seed)
    original_sample = policy._sample_actions
    captured = {}

    def record_sample(rng, observation, **kwargs):
        captured["sample_rng"] = np.asarray(jax.random.key_data(rng)).tolist()
        captured["model_inputs"] = {jax.tree_util.keystr(path): digest(value)
                                    for path, value in jax.tree_util.tree_flatten_with_path(observation)[0]}
        captured["arguments"] = (rng, observation, kwargs)
        return original_sample(rng, observation, **kwargs)

    policy._sample_actions = record_sample  # Read-only observation of the sole, original sampling call.
    print("Running the sole measured inference (includes JIT compilation)...", flush=True)
    start = time.monotonic()
    result = policy.infer(request)
    seconds = time.monotonic() - start
    arguments = captured.pop("arguments")
    np.savez_compressed(output / "actions.npz", actions=result["actions"])
    device = jax.devices()[0]
    gpu_libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "/" in line and any(name in line for name in
                                ["libcuda", "libcublas", "libcudnn", "libnvrtc", "libnvJitLink", "xla_cuda_plugin"])})
    report = {"scope": "Offline diagnostic; not an evaluation rollout",
              "fixture": "Recorded request" if args.request else "Deterministic integer RGB pattern v1, not an eval request",
              "profile_fingerprint": profile_fingerprint(), "config": POLICY_CONFIG,
              "python": sys.version, "packages": packages, "gpu": str(device.device_kind),
              "platform_version": str(device.client.platform_version),
              "compute_capability": str(getattr(device, "compute_capability", None)),
              "loaded_gpu_libraries": gpu_libraries, "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
              "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
              "xla_flags": os.environ.get("XLA_FLAGS"), "persistent_compilation_cache": jax.config.jax_enable_compilation_cache,
              "matmul_precision": str(jax.config.jax_default_matmul_precision), "enable_x64": jax.config.jax_enable_x64,
              "policy_seed": args.policy_seed, "infer_calls": 1, "warmup_calls": 0, "explicit_noise_supplied": False,
              "rng_after": np.asarray(jax.random.key_data(policy._rng)).tolist(),
              "request": {k: digest(v) for k, v in request.items()}, **captured,
              "checkpoint_files": weights, "source_files": sources, "seconds": seconds,
              "clean_actions": digest(result["actions"]), "clean_actions_values": np.asarray(result["actions"]).tolist()}
    tokenizer = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
    report["tokenizer_sha256"] = file_hash(tokenizer) if tokenizer.is_file() else None
    (output / "report.json").write_text(json.dumps(report, indent=2))
    if args.dump_hlo:
        print("Saving HLO for the measured model inputs; no extra inference...", flush=True)
        from openpi.shared.array_typing import disable_typechecking

        closure = inspect.getclosurevars(original_sample).nonlocals
        rng, observation, kwargs = arguments
        # JAX reconstructs Observation with ArgInfo placeholders during lowering.
        # Disable annotations only for this post-inference compiler inspection.
        with disable_typechecking():
            lowered = closure["jitted_fn"].lower(closure["state"], rng, observation, **kwargs)
        (output / "lowered.hlo.txt").write_text(lowered.compiler_ir("hlo").as_hlo_text())
        (output / "optimized.hlo.txt").write_text(lowered.compile().as_text())
        report["hlo_sha256"] = {p.name: file_hash(p) for p in output.glob("*.hlo.txt")}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file():
                bundle.write(path, arcname=path.name)
    print(json.dumps({"gpu": report["gpu"], "clean_actions": report["clean_actions"],
                      "sample_rng": report["sample_rng"], "seconds": seconds,
                      "report": str(output / "report.json"), "archive": str(archive)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
