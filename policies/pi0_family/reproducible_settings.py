"""Fixed local pi05 evaluation profile. This module does not import GPU packages."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "pi05-single-repro-v1"
POLICY_CONFIG = "pi05_droid_jointpos"
SCENE_SEED = 0
XLA_FLAGS = "--xla_gpu_autotune_level=0"
NRD_COMMIT = "4cdbf36c62913091b92cbfb08794f632320184f6"
NRD_LIBRARY_SHA256 = "2de390baf20bfe4e0728d7ad35f85ca942be548788c882ed7ae4cdc7fceb6f79"
SHIM_SOURCE_SHA256 = "7906ffa504701ebc56ab38fee353828c3210049e80134dad06f7827c7b24c29e"
HEADER_HASHES = {
    "NRD.h": "d1b430245330802f46752719754ec28f3b57d14aca674ce7628a8dab79b60f7f",
    "NRDDescs.h": "671d29b9be47c29a11a618c517beffa84f2cdcad68e64ee5f495570b7884c870",
    "NRDSettings.h": "cfa94b5198c08cee5b353fa3a502bcf7f80ec65193a9d639a9cb08a0e3747e29",
}
CACHE_SETTINGS = {
    "/rtx/raytracing/lightcache/spatialCache/enabled": False,
    "/rtx/raytracing/cached/enabled": False,
    "/rtx-transient/resourcemanager/enableTextureStreaming": False,
}
RENDER_SETTINGS = {
    "/rtx/post/aa/op": 0,
    "/rtx-transient/dldenoiser/enabled": False,
    "/rtx-transient/dlssg/enabled": False,
    "/rtx-transient/post/aa/limitedOps": False,
    **CACHE_SETTINGS,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile() -> dict:
    return {
        "id": PROFILE_ID, "num_envs": 1, "scene_seed": SCENE_SEED,
        "policy_config": POLICY_CONFIG, "policy_rng": "Seed once at process start; advance normally",
        "noise": "Original policy.infer(obs), no explicit noise, no warmup or repeat inference",
        "renderer": "RaytracedLighting", "render_settings": RENDER_SETTINGS,
        "nrd_constructor_seed": 0, "nrd_time_delta_ms": 66.67,
        "nrd_library_sha256": NRD_LIBRARY_SHA256, "shim_source_sha256": SHIM_SOURCE_SHA256,
        "xla_flags": XLA_FLAGS, "persistent_compilation_cache": False,
        "cameras": "Original TiledCameraCfg", "transport": "Original WebSocket server/client",
        "action_horizon": "Original client default; no override",
        "termination": "Original task success and timeout; no diagnostic step cap",
        "lifecycle": "Fresh policy and simulator processes for each task/seed entry",
    }


def profile_fingerprint() -> str:
    return hashlib.sha256(json.dumps(profile(), sort_keys=True).encode()).hexdigest()


def clean_environment() -> dict[str, str]:
    result = os.environ.copy()
    for key in list(result):
        if key.startswith(("ROBO_NRD", "ROBO_WATCH", "ROBO_ALLOC", "ROBO_STOCK", "ROBOLAB_REPRO_")):
            result.pop(key)
    # Diagnostic preload libraries must never leak into policy or simulator jobs.
    result.pop("LD_PRELOAD", None)
    result.update(OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
                  OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return result


def policy_environment(openpi_root: Path) -> dict[str, str]:
    result = clean_environment()
    result.update(PYTHONPATH=str(openpi_root / "src"), XLA_FLAGS=XLA_FLAGS,
                  XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false")
    return result


def simulator_environment(native: Path, policy_seed: int, run_index: int) -> dict[str, str]:
    result = clean_environment()
    result.update(PYTHONPATH=str(ROOT), LD_PRELOAD=str(native.resolve()),
                  ROBOLAB_REPRO_POLICY_SEED=str(policy_seed), ROBOLAB_REPRO_RUN_INDEX=str(run_index))
    return result


def inspect_simulator(python: Path) -> dict:
    script = """
import hashlib,json,sysconfig
from pathlib import Path
from importlib.metadata import version
root=Path(sysconfig.get_paths()['purelib'])/'isaacsim'
libs=list(root.glob('extscache/omni.hydra.rtx-*/bin/deps/libNRD.so.3.2.3'))
print(json.dumps({'isaacsim':version('isaacsim'),'nrd_libraries':[
 {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in libs]}))
"""
    info = json.loads(subprocess.check_output([str(python), "-c", script], text=True, env=clean_environment()))
    if info["isaacsim"] != "5.0.0.0":
        raise RuntimeError(f"{PROFILE_ID} requires the verified Isaac Sim 5.0.0.0 stack: {info}")
    libs = info["nrd_libraries"]
    if len(libs) != 1 or libs[0]["sha256"] != NRD_LIBRARY_SHA256:
        raise RuntimeError(f"NRD binary does not match the verified ABI; refusing to run: {info}")
    return info


def build_native() -> Path:
    """Build the exact verified shim; cache pinned SDK headers outside version control."""
    cache = ROOT / ".cache/reproducible_eval"
    include = cache / "nrd/include"
    include.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("native") / "nrd_controls.cpp"
    if sha256(source) != SHIM_SOURCE_SHA256:
        raise RuntimeError("NRD control source differs from the verified profile; review before changing the profile")
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, expected in HEADER_HASHES.items():
            target = include / name
            if not target.exists():
                url = f"https://raw.githubusercontent.com/NVIDIA-RTX/NRD/{NRD_COMMIT}/Include/{name}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise RuntimeError(f"Unexpected NRD SDK header: {name}")
                target.write_bytes(content)
            if sha256(target) != expected:
                raise RuntimeError(f"Cached NRD SDK header hash mismatch: {target}")
        compiler = shutil.which("c++")
        if not compiler:
            raise RuntimeError("A C++17 compiler is required to build the NRD controls")
        compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
        key = hashlib.sha256((SHIM_SOURCE_SHA256 + compiler_version).encode()).hexdigest()[:16]
        output = cache / "build" / key / "libnrd_controls.so"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            temporary = output.with_suffix(f".{os.getpid()}.tmp.so")
            try:
                subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-shared", str(source),
                                "-I", str(include), "-o", str(temporary), "-ldl", "-pthread"], check=True)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        return output


def apply_render_config(cfg):
    cfg.sim.render.antialiasing_mode = "Off"
    cfg.sim.render.enable_dl_denoiser = False
    cfg.sim.render.enable_dlssg = False
    cfg.sim.render.carb_settings = {**(cfg.sim.render.carb_settings or {}), **CACHE_SETTINGS}
    return cfg


def verify_server_metadata(metadata: dict, policy_seed: int) -> None:
    actual = metadata.get("robolab_reproducible", {})
    expected = {"profile": profile_fingerprint(), "policy_seed": policy_seed, "config": POLICY_CONFIG}
    if any(actual.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Policy server is not running the requested reproducible profile: {actual}")
