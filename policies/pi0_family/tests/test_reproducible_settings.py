"""CPU-only checks for isolation, seed lifecycle, and rejecting mixed profiles."""
import importlib.util
import json
import os
from pathlib import Path
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
                                     "ROBO_WATCH_DEBUG": "1", "CUDA_VISIBLE_DEVICES": "2"}):
            env = settings.policy_environment(Path("/tmp/openpi"))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("ROBO_NRD_LOG", env)
        self.assertNotIn("ROBO_WATCH_DEBUG", env)
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


if __name__ == "__main__":
    unittest.main()
