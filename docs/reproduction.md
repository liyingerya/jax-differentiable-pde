# Reproduction guide

## Validated environment

Python 3.10.8; packages: {'jax': '0.6.2', 'jaxlib': '0.6.2', 'numpy': '2.2.6', 'pytest': '9.1.1', 'matplotlib': '3.10.9'}. CPU/macOS hardware details for archived timing are in [Stage 11](stage11_results.json), `environment`. This run uses the existing CPU environment. `requirements.txt` retains the project policy of unpinned direct dependencies: JAX, pytest, matplotlib. NumPy is supplied by JAX. No pandas, seaborn or additional ML framework was added. Exact future-version reproducibility is not promised; the versions here identify the validated environment.

## Quick reproduction — no heavy studies

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m examples.quickstart
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.final_figures
```

The quickstart evaluates a loss and gradient, not optimization. Its warmed execution is below a few seconds on the validated machine. Tests take about 36 seconds here; figure rendering takes seconds after font-cache setup. Figure data come only from archived JSON. To regenerate documentation/audits, capture pytest to `docs/stage12_pytest.txt`, then run `python -m experiments.final_synthesis`. The synthesis script checks the pre-existing `docs/stage12_preservation.json`; do not replace that baseline to hide a mismatch.

## Full reproduction — separate disposable workspace

Historical experiment drivers can overwrite their own outputs. Preserve the original repository and run them only in a separate copy. Stage 11's archive guard compares the entire earlier file inventory; adding final documents makes that old inventory guard intentionally reject the synthesis tree. Use `python -m experiments.reproduction_workspace --destination /tmp/jax-pde-stage11-reproduction` to export the exact archived Stage 11 file set, without venv/caches/final documents. Install requirements in that destination. This exporter copies files and performs no scientific work.

Existing checkpoints normally resume rather than rerun completed groups. To recompute a stage, use another disposable copy and explicitly move that stage's output/checkpoint aside after reading its driver. Keep dependency-stage archives intact. If upstream records are regenerated, strict downstream byte-hash guards can reject them even when numeric differences are only timing/roundoff; investigate and compare values rather than disabling a guard. This release does not claim a one-command fresh run of all heavy archives.

Historical commands, run from a suitable disposable workspace:

```sh
python -m experiments.stage1_derivative_validation
python -m experiments.stage2_time_integration_validation
python -m experiments.stage3_parameter_inference
python -m experiments.stage4_refinement_study
python -m experiments.stage5_observation_robustness
python -m experiments.stage6_joint_inference
python -m experiments.stage7_observation_design --design-only
python -m experiments.stage7_observation_design --recover-frozen
python -m experiments.stage7_observation_design --harden-existing
python -m experiments.stage8_robust_design --design-only
python -m experiments.stage8_robust_design --recover-frozen
```

Stage 9 phases:

```sh
for phase in forward-validation calibrate clean-inference noisy-inference finalize; do
  python -m experiments.stage9_model_discrepancy --"$phase" || break
done
```

Stage 10 phases (HEAVY):

```sh
for phase in clean-profiles profile-surfaces bootstrap-moderate bootstrap-severe heldout-transfer matched-controls representative-profiles finalize; do
  python -m experiments.stage10_identifiability --"$phase" || break
done
```

Stage 11 phases:

```sh
python -m pytest -q > docs/stage11_pytest.txt
for phase in symbols temporal stability gradients inverse benchmark accuracy finalize; do
  python -m experiments.stage11_scalable_integration --phase "$phase" || break
done
```

## Computational burden

Stages 1–2 are small validations; Stages 3–7 involve repeated differentiable solves and can take minutes or more depending on hardware. Stage 8 is **heavy**: 14056 candidate combinations per budget, scenario sensitivities and repeated recovery. Stage 9 has 527 archived production fits plus diagnostics. Stage 10 is **very heavy**: 48336 archived optimizer attempts including profile grids and repeated-noise studies; reserve a long checkpointed run rather than expecting a quick demo. These counts describe archived work, not newly measured wall-clock estimates. Stage 11 skips excessive fine-grid scan workloads by declared caps. Consult each [stage report](stage_index.md) before execution. Stage 12 reruns none of these production studies.

## Artifact policy

Figures are generated deterministically from saved data with fixed styles and SVG metadata. Same-environment byte reproducibility is audited; fonts/library changes can change rendering. `docs/release_manifest.txt` lists intended deliverables and excludes virtual environments, Python/pytest/JAX caches and OS metadata. Do not package the local `.venv`.
