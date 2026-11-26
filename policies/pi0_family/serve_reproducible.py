"""Original OpenPI WebSocket policy serving with fixed startup seed/XLA configuration."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

from reproducible_settings import POLICY_CONFIG, ROOT, policy_environment, profile_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=ROOT.parent / "openpi-robolab")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 0 <= args.policy_seed < 2**32:
        parser.error("policy seed must be in [0, 2**32)")
    # These are set before importing JAX/OpenPI. No policy preload is allowed.
    environment = policy_environment(args.openpi_root.resolve())
    for key in list(os.environ):
        if key == "LD_PRELOAD" or key.startswith(("ROBO_NRD", "ROBO_WATCH", "ROBO_ALLOC")):
            os.environ.pop(key, None)
    os.environ.update(environment)
    sys.path.insert(0, str(args.openpi_root.resolve() / "src"))
    import jax
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    spec = importlib.util.spec_from_file_location("official_serve_policy", args.openpi_root / "scripts/serve_policy.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the original OpenPI serve_policy.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    official_args = official.Args(port=args.port, policy=official.Checkpoint(
        config=POLICY_CONFIG, dir=args.checkpoint))
    policy = official.create_policy(official_args)
    if policy._is_pytorch_model:
        raise RuntimeError("This profile was verified with the JAX pi05_droid_jointpos policy")
    if policy._sample_kwargs:
        raise RuntimeError("Unexpected sampling overrides")
    policy._rng = jax.random.key(args.policy_seed)
    # No dummy observations, warmup calls, explicit noise, or infer wrappers.
    metadata = {**policy.metadata, "robolab_reproducible": {
        "profile": profile_fingerprint(), "policy_seed": args.policy_seed, "config": POLICY_CONFIG}}
    print(json.dumps({"event": "reproducible_policy_ready", "pid": os.getpid(), **metadata}), flush=True)
    logging.basicConfig(level=logging.INFO, force=True)
    WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
