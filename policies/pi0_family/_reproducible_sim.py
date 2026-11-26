"""Original RoboLab CLI with the fixed render profile; no episode/action overrides."""
import json
import os
from pathlib import Path
import runpy

from reproducible_settings import (
    CACHE_SETTINGS, RENDER_SETTINGS, ROOT, SCENE_SEED, apply_render_config,
    probe_native_preload, profile, profile_fingerprint, verify_server_metadata,
)

# Fail before constructing SimulationApp if a binary launcher discarded the shim.
native_preload_check = probe_native_preload()

from robolab.eval import runner

original_driver = runner.run_evaluation


def fixed_driver(args, *, policy, client_factory, **kwargs):
    import carb
    from robolab.core.environments import runtime
    if (args.num_envs != 1 or args.num_runs != 1 or args.seed != SCENE_SEED
            or policy != "pi05" or args.open_loop_horizon is not None
            or args.init_pose_noise is not None or args.randomize_background
            or args.renderer != "realtime" or args.rendering_type is not None
            or args.num_episodes_adaptive is not None):
        raise RuntimeError("Arguments conflict with the fixed reproducible evaluation profile")
    native = Path(native_preload_check["native"])
    settings = carb.settings.get_settings()
    if any(settings.get(k) != v for k, v in CACHE_SETTINGS.items()):
        raise RuntimeError("Startup cache/streaming settings were not applied before scene creation")
    original_parse = runtime.parse_env_cfg

    def fixed_parse(*a, **k):
        return apply_render_config(original_parse(*a, **k))

    runtime.parse_env_cfg = fixed_parse
    seed = int(os.environ["ROBOLAB_REPRO_POLICY_SEED"])

    def factory(cli_args):
        # The original client creates its connection and owns its chunk/RNG-free state.
        client = client_factory(cli_args)
        verify_server_metadata(client.client.get_server_metadata(), seed)
        actual = {k: settings.get(k) for k in RENDER_SETTINGS}
        if actual != RENDER_SETTINGS:
            raise RuntimeError(f"Render profile mismatch: {actual}")
        output = Path(args.output_folder_name).resolve() / args.task[0]
        output.mkdir(parents=True, exist_ok=True)
        (output / f"run_{args.run_index_offset}.reproducible.json").write_text(json.dumps({
            "profile": profile(), "profile_fingerprint": profile_fingerprint(),
            "scene_seed": SCENE_SEED, "policy_seed": seed, "native_preload": str(native),
            "native_preload_check": native_preload_check,
            "simulator_pid": os.getpid(), "actual_render_settings": actual,
            "diagnostic_step_cap": None, "extra_inference_calls": 0,
        }, indent=2))
        return client

    # No custom episode_runner, env wrapper, state polling, or extra policy calls.
    return original_driver(args, policy=policy, client_factory=factory, **kwargs)


runner.run_evaluation = fixed_driver
runpy.run_path(str(ROOT / "policies/pi0_family/run.py"), run_name="__main__")
