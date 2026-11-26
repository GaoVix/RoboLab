"""CPU-only checks for isolation, seed lifecycle, and rejecting mixed profiles."""
import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import reproducible_settings as settings
import run_reproducible as launcher
import serve_reproducible as serving


class ReproducibleSettingsTests(unittest.TestCase):
    def test_policy_process_never_inherits_native_diagnostic_preloads(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "/tmp/old_probe.so", "ROBO_NRD_LOG": "/tmp/old.log",
                                     "ROBO_WATCH_DEBUG": "1", "CUDA_VISIBLE_DEVICES": "2",
                                     "PYTHONEXE": "/tmp/old_wrapper",
                                     "ROBOLAB_REPRO_NATIVE": "/tmp/old_probe.so"}):
            env = settings.policy_environment(Path("/tmp/openpi"))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("ROBO_NRD_LOG", env)
        self.assertNotIn("ROBO_WATCH_DEBUG", env)
        self.assertNotIn("PYTHONEXE", env)
        self.assertNotIn("ROBOLAB_REPRO_NATIVE", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["XLA_FLAGS"], "--xla_gpu_autotune_level=0")

    def test_renderer_profile_does_not_replace_cameras_actions_or_terminations(self):
        cameras, actions, terminations = object(), object(), object()
        cfg = types.SimpleNamespace(
            scene=cameras, actions=actions, terminations=terminations, episode_length_s=180,
            sim=types.SimpleNamespace(render=types.SimpleNamespace(carb_settings={"unrelated": True})))
        settings.apply_render_config(cfg)
        self.assertIs(cfg.scene, cameras)
        self.assertIs(cfg.actions, actions)
        self.assertIs(cfg.terminations, terminations)
        self.assertEqual(cfg.episode_length_s, 180)
        self.assertTrue(cfg.sim.render.carb_settings["unrelated"])

    def test_wrong_server_or_seed_is_rejected(self):
        with self.assertRaises(RuntimeError):
            settings.verify_server_metadata({}, 0)
        metadata = {"robolab_reproducible": {"profile": settings.profile_fingerprint(),
                    "policy_seed": 7, "config": settings.POLICY_CONFIG}}
        settings.verify_server_metadata(metadata, 7)
        with self.assertRaises(RuntimeError):
            settings.verify_server_metadata(metadata, 0)

    def test_other_isaac_abi_is_rejected_before_launch(self):
        result = json.dumps({"isaacsim": "6.0.0", "nrd_libraries": []})
        with patch.object(settings.subprocess, "check_output", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "requires the verified Isaac Sim"):
                settings.inspect_simulator(Path("/tmp/python"))

    def test_pip_and_docker_layouts_resolve_same_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "isaac-sim"
            lib = root / "extscache/omni.hydra.rtx-test/bin/deps/libNRD.so.3.2.3"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"NRD fixture")
            pip = settings._probe_simulator_installation([root], "5.0.0.0")
            (root / "VERSION").write_text("5.0.0-rc.45+release.23960.184afb15.gl\n")
            alias = Path(directory) / "_isaac_sim"
            alias.symlink_to(root, target_is_directory=True)
            docker = settings._probe_simulator_installation([root, alias])
            self.assertEqual(pip, docker)
            self.assertEqual(docker["isaacsim"], "5.0.0.0")
            self.assertEqual(len(docker["nrd_libraries"]), 1)
            self.assertEqual(docker["nrd_libraries"][0]["sha256"],
                             hashlib.sha256(b"NRD fixture").hexdigest())

    def test_docker_version_does_not_bypass_exact_nrd_hash_check(self):
        result = json.dumps({"isaacsim": "5.0.0.0", "nrd_libraries": [
            {"path": "/isaac-sim/extscache/libNRD.so.3.2.3", "sha256": "wrong"}]})
        with patch.object(settings.subprocess, "check_output", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "does not match the verified ABI"):
                settings.inspect_simulator(Path("/tmp/python.sh"))

    def test_binary_isaac6_version_remains_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib = root / "extscache/omni.hydra.rtx-test/bin/deps/libNRD.so.3.2.3"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"NRD fixture")
            (root / "VERSION").write_text("6.0.1+release\n")
            result = settings._probe_simulator_installation([root])
            with patch.object(settings.subprocess, "check_output", return_value=json.dumps(result)):
                with self.assertRaisesRegex(RuntimeError, "requires the verified Isaac Sim"):
                    settings.inspect_simulator(Path("/tmp/python.sh"))

    def test_public_entry_does_not_accept_multi_env_override(self):
        result = subprocess.run([sys.executable, str(HERE / "run_reproducible.py"),
                                 "--num-envs", "2", "--check"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_resume_requires_matching_task_and_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertFalse(launcher.completed(out, "A", 0))
            (out / "episode_results.jsonl").write_text(json.dumps({
                "env_name": "A", "episode": 1, "success": False}) + "\n")
            self.assertFalse(launcher.completed(out, "A", 0))
            self.assertFalse(launcher.completed(out, "B", 1))
            self.assertTrue(launcher.completed(out, "A", 1))

    def test_server_seeds_once_without_any_inference_or_warmup(self):
        assignments, calls, servers = [], [], []
        class FakePolicy:
            _is_pytorch_model = False
            _sample_kwargs = {}
            metadata = {}
            def __setattr__(self, name, value):
                if name == "_rng":
                    assignments.append(value)
                object.__setattr__(self, name, value)
            def infer(self, *a, **k):
                calls.append((a, k))
                raise AssertionError("Serving startup must not perform any inference")
        policy = FakePolicy()
        class FakeServer:
            def __init__(self, **kwargs):
                servers.append(kwargs)
            def serve_forever(self):
                pass
        official = types.ModuleType("official_serve_policy")
        official.Args = lambda **k: types.SimpleNamespace(**k)
        official.Checkpoint = lambda **k: types.SimpleNamespace(**k)
        official.create_policy = lambda args: policy
        spec = types.SimpleNamespace(name="official_serve_policy",
                                     loader=types.SimpleNamespace(exec_module=lambda module: None))
        fake_jax = types.ModuleType("jax")
        fake_jax.random = types.SimpleNamespace(key=lambda seed: ("key", seed))
        fake_ws = types.ModuleType("openpi.serving.websocket_policy_server")
        fake_ws.WebsocketPolicyServer = FakeServer
        with patch.dict(os.environ), patch.dict(sys.modules, {
            "jax": fake_jax, "openpi": types.ModuleType("openpi"),
            "openpi.serving": types.ModuleType("openpi.serving"),
            "openpi.serving.websocket_policy_server": fake_ws,
        }), patch.object(sys, "argv", ["server", "--checkpoint", "/tmp/checkpoint", "--policy-seed", "7"]), \
             patch.object(importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(importlib.util, "module_from_spec", return_value=official):
            serving.main()
        self.assertEqual(assignments, [("key", 7)])
        self.assertEqual(calls, [])
        self.assertIs(servers[0]["policy"], policy)
        settings.verify_server_metadata(servers[0]["metadata"], 7)


class NativeLaunchTests(unittest.TestCase):
    """Exercise ELF loading through a binary launcher which overwrites LD_PRELOAD."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler needed for native launch regression checks")
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.root = Path(cls.fixture.name)
        cls.library = cls.root / "controls.so"
        subprocess.run([compiler, "-shared", "-fPIC", "-x", "c++", "-", "-o", str(cls.library)],
                       input='extern "C" void GetComputeDispatches() {}\nextern "C" int rand() noexcept { return 0; }\n',
                       text=True, capture_output=True, check=True)
        cls.competing = cls.root / "competing.so"
        shutil.copyfile(cls.library, cls.competing)
        binary = cls.root / "kit/python/bin/python3"
        binary.parent.mkdir(parents=True)
        binary.symlink_to(sys.executable)
        cls.wrapper = cls.root / "python.sh"
        cls.wrapper.write_text(
            '#!/bin/bash\nset -e\n'
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'python_exe=${PYTHONEXE:-"${SCRIPT_DIR}/kit/python/bin/python3"}\n'
            'export LD_PRELOAD=libc.so.6\n'
            '$python_exe "$@"\n')
        cls.wrapper.chmod(0o755)

    def test_binary_launcher_overwrite_is_reproduced_and_fixed(self):
        script = inspect.getsource(settings.probe_native_preload) + "\nprobe_native_preload()\n"
        # Same failure mode as the official python.sh, before enabling its hook.
        broken = subprocess.run([str(self.wrapper), "-c", script], text=True, capture_output=True,
                                env=settings.simulator_environment(self.library, 0, 0))
        self.assertNotEqual(broken.returncode, 0)
        self.assertIn("NRD control preload is not mapped", broken.stderr)
        checked = settings.inspect_native_preload(self.wrapper, self.library)
        self.assertEqual(checked["symbols"], {
            "GetComputeDispatches": str(self.library), "rand": str(self.library)})
        self.assertEqual(checked["ld_preload"], f"{self.library}:libc.so.6")

    def test_direct_python_launch_also_checks_real_bindings(self):
        checked = settings.inspect_native_preload(Path(sys.executable), self.library)
        self.assertEqual(checked["symbols"]["GetComputeDispatches"], str(self.library))
        self.assertEqual(checked["symbols"]["rand"], str(self.library))
        self.assertEqual(checked["ld_preload"], str(self.library))

    def test_mapped_library_with_wrong_global_symbol_binding_is_rejected(self):
        script = inspect.getsource(settings.probe_native_preload) + "\nprobe_native_preload()\n"
        env = settings.simulator_environment(self.library, 0, 0)
        env["LD_PRELOAD"] = f"{self.competing}:{self.library}"
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"resolves to {self.competing}", result.stderr)

    def test_binary_launcher_missing_bundled_python_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Required simulator launch executable is missing"):
            settings.simulator_environment(self.library, 0, 0, python=self.root / "missing/python.sh")


if __name__ == "__main__":
    unittest.main()
