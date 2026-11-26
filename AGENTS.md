# Evaluation defaults for this workspace

The user requires the verified reproducible pi0.5 evaluation protocol for future
evaluations. Use `policies/pi0_family/run_reproducible.py` as the default entry
point. Do not silently fall back to the original runner with different settings.

- One env per simulator process. Start fresh policy and simulator processes for
  every task/seed entry. `--policy-seeds` lists each episode's policy seed;
  repeated seeds mean repeated independent launches with that same seed.
- Seed the policy RNG once before the first real request. Preserve the original
  `policy.infer(obs)` and its fresh noise sampling on every request. Never inject
  a fixed noise tensor, reset RNG per inference, or add fake warmups/repeat calls.
- Preserve original pi05 action chunking, preprocessing, WebSocket transport,
  camera classes, `env.step`, success conditions, timeout, and recorder behavior.
- Apply the fixed render/NRD/XLA profile in `reproducible_settings.py`. Do not
  omit the native NRD controls or disable ABI/version checks to make a run start.
- Full evaluations use the task's original step budget and termination logic.
  Short diagnostic caps are allowed only when explicitly requested and must be
  labeled as diagnostics; they are not full task SR results.
- Default scene seed is 0, initial-pose/background randomization is off. Default
  videos are off; numeric official HDF5/results remain enabled.
- Record seeds/profile with results. Do not promise bitwise equality across
  different hardware/software stacks or all tasks based on one test.

Evidence: on 2026-11-26, two complete independent restarts of the original
WebSocket policy/simulator flow for ClutterPlasticTask, single env, seed 0,
600 steps each, produced identical 40 query inputs/clean chunks and all 132
official recorded datasets. The full-task entry has no 600-step cap. The original
runner is retained for upstream comparison, not as the default for our evals.
