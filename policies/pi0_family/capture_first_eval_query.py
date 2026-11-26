"""Capture the first real eval request/response and stop before applying any action.

Diagnostic only: no rollout result or success rate is produced. Scene creation,
resets, cameras, request packing, policy RNG and WebSocket inference are original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

from reproducible_settings import (
    CACHE_SETTINGS, ROOT, SCENE_SEED, build_native, inspect_native_preload, inspect_simulator,
    policy_environment, profile, simulator_environment,
)


def simulator_child(output: Path, cli_args: list[str]):
    from robolab.eval import runner

    original_driver = runner.run_evaluation

    class FirstQueryCaptured(Exception):
        pass

    def diagnostic_driver(args, *, client_factory, **kwargs):
        def recording_factory(cli):
            client = client_factory(cli)
            original_query = client._query_server

            def query(request):
                from importlib.metadata import PackageNotFoundError, version
                import numpy as np

                np.savez_compressed(output / "request.npz", **{
                    k.replace("/", "__"): v for k, v in request.items()})
                response = original_query(request)
                np.savez_compressed(output / "clean_actions.npz", actions=response["actions"])
                packages = {}
                for name in ["isaaclab", "isaaclab_tasks", "isaacsim", "warp-lang", "torch", "numpy", "opencv-python", "pillow"]:
                    try:
                        packages[name] = version(name)
                    except PackageNotFoundError:
                        packages[name] = None
                (output / "capture.json").write_text(json.dumps({
                    "scope": "First real policy request and response only; not a rollout or SR result",
                    "client_query_calls": 1, "applied_control_steps": 0,
                    "policy_seed": 0, "scene_seed": SCENE_SEED, "profile": profile(),
                    "request_keys": sorted(request), "simulator_packages": packages,
                }, indent=2))
                # Normal episode/runner finally blocks close the client and env.
                # No fake observations or actions are returned to the control loop.
                raise FirstQueryCaptured

            client._query_server = query
            return client

        try:
            return original_driver(args, client_factory=recording_factory, **kwargs)
        except FirstQueryCaptured:
            print("Captured first real request/response; stopped before applying an action.", flush=True)

    runner.run_evaluation = diagnostic_driver
    bootstrap = Path(__file__).with_name("_reproducible_sim.py")
    sys.argv = [str(bootstrap), *cli_args]
    runpy.run_path(str(bootstrap), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--openpi-root", type=Path, default=ROOT.parent / "openpi-robolab")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / ".cache/openpi/openpi-assets-simeval/pi05_droid_jointpos")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["request.npz", "clean_actions.npz", "capture.json"]):
        parser.error("Capture already exists; use a fresh diagnostic output directory")
    stack = inspect_simulator(args.sim_python)
    native = build_native()
    preload = inspect_native_preload(args.sim_python, native)
    (output / "installation.json").write_text(json.dumps({"simulator_stack": stack, "preload": preload}, indent=2))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    here = Path(__file__).resolve().parent
    policy = None
    try:
        with (output / "policy.log").open("w") as log:
            policy = subprocess.Popen([
                str(args.openpi_root / ".venv/bin/python"), "-u", str(here / "serve_reproducible.py"),
                "--openpi-root", str(args.openpi_root), "--checkpoint", str(args.checkpoint),
                "--policy-seed", "0", "--port", str(port),
            ], cwd=args.openpi_root, env=policy_environment(args.openpi_root), stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while True:
                if policy.poll() is not None:
                    raise RuntimeError(f"Policy startup failed; see {output / 'policy.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy startup")
                    time.sleep(1)
            print("Capturing one real ClutterPlasticTask request, seed 0; zero applied control steps...", flush=True)
            with (output / "sim.log").open("w") as log:
                result = subprocess.run([
                    str(args.sim_python), "-u", str(Path(__file__).resolve()), "--capture-sim", str(output),
                    "--policy", "pi05", "--task", "ClutterPlasticTask", "--num-envs", "1", "--num-runs", "1",
                    "--run-index-offset", "0", "--seed", str(SCENE_SEED), "--headless", "--video-mode", "none",
                    "--remote-host", "127.0.0.1", "--remote-port", str(port),
                    "--output-folder-name", str(output / "diagnostic_only"),
                    "--kit_args", " ".join("--" + key + "=false" for key in CACHE_SETTINGS),
                ], cwd=ROOT, env=simulator_environment(native, 0, 0, python=args.sim_python),
                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not (output / "capture.json").exists():
                raise RuntimeError(f"Capture failed (exit {result.returncode}); see {output / 'sim.log'}")
    finally:
        from run_reproducible import stop

        stop(policy)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ["request.npz", "clean_actions.npz", "capture.json", "installation.json", "policy.log", "sim.log"]:
            bundle.write(output / name, arcname=name)
        task_output = output / "diagnostic_only/ClutterPlasticTask"
        for name, archive_name in [("env_cfg.json", "env_cfg.json"),
                                   ("run_0.reproducible.json", "render_profile.json")]:
            path = task_output / name
            if path.is_file():
                bundle.write(path, arcname=archive_name)
    print(f"First query captured; applied control steps: 0. Archive: {archive}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-sim":
        simulator_child(Path(sys.argv[2]), sys.argv[3:])
    else:
        main()
