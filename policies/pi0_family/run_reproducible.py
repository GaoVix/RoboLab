"""Run full pi05 episodes with the fixed single-env reproducibility profile.

Each explicit policy-seed entry gets fresh policy and simulator processes.
The original RoboLab runner handles task limits, success, actions and recording.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from reproducible_settings import (
    CACHE_SETTINGS, ROOT, SCENE_SEED, build_native, inspect_simulator,
    policy_environment, profile, profile_fingerprint, sha256, simulator_environment,
)


def task_names(requested: list[str] | None) -> list[str]:
    rows = json.loads((ROOT / "robolab/tasks/_metadata/task_metadata.json").read_text())
    names = [row["task_name"] for row in rows if row["filename"].startswith("benchmark/")]
    if requested is None:
        return names
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"Unknown benchmark tasks: {sorted(unknown)}")
    return list(dict.fromkeys(requested))


def completed(output: Path, task: str, index: int) -> bool:
    path = output / "episode_results.jsonl"
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("env_name") == task and row.get("episode") == index:
                return True
    return False


def stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", default=None, help="Benchmark tasks; default: all benchmark tasks")
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[0],
                        help="One fresh episode per explicit seed, in order (default: 0). Repeated seeds are allowed.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--openpi-root", type=Path, default=ROOT.parent / "openpi-robolab")
    parser.add_argument("--sim-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--policy-python", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / ".cache/openpi/openpi-assets-simeval/pi05_droid_jointpos")
    parser.add_argument("--video-mode", choices=["none", "all", "sensor", "viewport"], default="none")
    parser.add_argument("--check", action="store_true", help="Build/check the profile and print the plan without starting GPU jobs")
    args = parser.parse_args()
    if any(not 0 <= seed < 2**32 for seed in args.policy_seeds):
        parser.error("policy seeds must be in [0, 2**32)")
    try:
        tasks = task_names(args.task)
    except ValueError as error:
        parser.error(str(error))
    openpi_root = args.openpi_root.resolve()
    sim_python = args.sim_python.absolute()
    # Do not resolve venv Python symlinks: the path identifies the environment.
    policy_python = (args.policy_python or openpi_root / ".venv/bin/python").absolute()
    checkpoint = args.checkpoint.resolve()
    for path in [sim_python, policy_python, openpi_root / "scripts/serve_policy.py", checkpoint / "params"]:
        if not path.exists():
            parser.error(f"Required path does not exist: {path}")
    stack = inspect_simulator(sim_python)
    native = build_native()
    plan = {"profile": profile(), "profile_fingerprint": profile_fingerprint(),
            "tasks": tasks, "policy_seeds": args.policy_seeds, "checkpoint": str(checkpoint),
            "sim_python": str(sim_python), "policy_python": str(policy_python),
            "openpi_root": str(openpi_root), "simulator_stack": stack,
            "native_preload": str(native), "native_preload_sha256": sha256(native),
            "video_mode": args.video_mode, "image_recording": False}
    if args.check:
        print(json.dumps({**plan, "tasks": f"{len(tasks)} selected", "gpu_jobs_started": 0}, indent=2))
        return
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or ROOT / "output" / f"repro_pi05_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "reproducible_protocol.json"
    if protocol_file.exists():
        previous = json.loads(protocol_file.read_text())
        for key in ["profile_fingerprint", "tasks", "policy_seeds", "checkpoint", "simulator_stack"]:
            if previous.get(key) != plan[key]:
                raise RuntimeError(f"Existing output uses different {key}; use a new output directory")
    else:
        protocol_file.write_text(json.dumps(plan, indent=2))
    print(f"Reproducible eval: {len(tasks)} tasks × {len(args.policy_seeds)} episodes; output: {output}", flush=True)
    here = Path(__file__).resolve().parent
    for task in tasks:
        for index, seed in enumerate(args.policy_seeds):
            if completed(output, task, index):
                print(f"Skip completed {task} episode {index} (policy seed {seed})", flush=True)
                continue
            directory = output / task
            directory.mkdir(exist_ok=True)
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            policy = sim = None
            life = {"task": task, "episode": index, "policy_seed": seed,
                    "scene_seed": SCENE_SEED, "start_monotonic": time.monotonic(),
                    "fresh_processes": True}
            try:
                with (directory / f"run_{index}.policy.log").open("w") as plog, \
                     (directory / f"run_{index}.sim.log").open("w") as slog:
                    policy = subprocess.Popen([
                        str(policy_python), "-u", str(here / "serve_reproducible.py"),
                        "--openpi-root", str(openpi_root), "--checkpoint", str(checkpoint),
                        "--policy-seed", str(seed), "--port", str(port),
                    ], cwd=openpi_root, env=policy_environment(openpi_root), stdout=plog, stderr=subprocess.STDOUT)
                    life["policy_pid"] = policy.pid
                    deadline = time.monotonic() + 600
                    while True:
                        if policy.poll() is not None:
                            raise RuntimeError(f"Policy failed; see {directory / f'run_{index}.policy.log'}")
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                                if response.status == 200:
                                    break
                        except (OSError, urllib.error.URLError):
                            if time.monotonic() > deadline:
                                raise TimeoutError("Policy server startup")
                            time.sleep(1)
                    kit_args = " ".join("--" + key + "=false" for key in CACHE_SETTINGS)
                    sim = subprocess.Popen([
                        str(sim_python), "-u", str(here / "_reproducible_sim.py"),
                        "--policy", "pi05", "--task", task, "--num-envs", "1", "--num-runs", "1",
                        "--run-index-offset", str(index), "--seed", str(SCENE_SEED), "--headless",
                        "--video-mode", args.video_mode, "--remote-host", "127.0.0.1", "--remote-port", str(port),
                        "--output-folder-name", str(output), "--kit_args", kit_args,
                    ], cwd=ROOT, env=simulator_environment(native, seed, index), stdout=slog, stderr=subprocess.STDOUT)
                    life["simulator_pid"] = sim.pid
                    print(f"Run {task}, episode {index}, policy seed {seed}, one env", flush=True)
                    last_notice = time.monotonic()
                    while sim.poll() is None:
                        if policy.poll() is not None:
                            raise RuntimeError("Policy server exited during evaluation")
                        if time.monotonic() - last_notice >= 60:
                            print(f"Running {task}, episode {index}: {int(time.monotonic()-life['start_monotonic'])} s", flush=True)
                            last_notice = time.monotonic()
                        time.sleep(2)
                    life["simulator_exit_code"] = sim.returncode
                    if sim.returncode:
                        raise RuntimeError(f"Simulator failed; see {directory / f'run_{index}.sim.log'}")
                    if not completed(output, task, index):
                        raise RuntimeError("Original evaluator exited without a completed episode result")
            finally:
                stop(sim)
                stop(policy)
                life.update(end_monotonic=time.monotonic(),
                            policy_exit_code=policy.returncode if policy is not None else None,
                            simulator_exit_code=sim.returncode if sim is not None else None)
                (directory / f"run_{index}.lifecycle.json").write_text(json.dumps(life, indent=2))
            print(f"Completed {task}, episode {index}; both processes exited", flush=True)


if __name__ == "__main__":
    main()
