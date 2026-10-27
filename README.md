# Dataset and code: decision-focused chassis world models

Public dataset and companion-code repository for **Physics-Structured Chassis World Models with Decision-Focused Tire-Force Learning**.

This repository preserves the study's physics, force-residual networks, learning scripts, planners, adapted neural-ExpTanh comparator and saved evaluation evidence. The source copies retain their original versioned filenames and hashes. New release utilities configure a local licensed parameter asset and verify the downloaded bytes.

## Contents

- `trfc_drift/`, `trfc_wm/` and `tools/`: original implementation and its statically resolved local Python imports.
- `checkpoints/`: 22 trained checkpoints, including the forecast baselines, selected residual, independent replicas and fitted ExpTanh variants.
- `experiments/results/`: saved development, confirmation and independent-checker JSON files. Courses, seeds and settings remain in the original receipts.
- `specification/`: implementation supplement source, full-precision physical coefficients and training-configuration JSON.
- `manifests/`: source/checkpoint hashes, data-asset hashes, original-to-release checkpoint mapping and licensed-asset identification.
- `release_tools/`: byte verification, a CPU-only import/checkpoint smoke, and an offline script launcher.

## Numerical data assets

Download the numerical data from the [v1.0.0 dataset release](https://github.com/das1iw/decision-focused-chassis-world-models/releases/tag/v1.0.0). See [DATASET_CARD.md](DATASET_CARD.md) for provenance, fields, splits and limitations. [CITATION.cff](CITATION.cff) identifies this repository as the dataset associated with the paper.

| Asset | Bytes | Installation |
|---|---:|---|
| [sequence_data_v1.npz](https://github.com/das1iw/decision-focused-chassis-world-models/releases/download/v1.0.0/sequence_data_v1.npz) | 76,933,168 | `data/sequence_data_v1.npz` |
| [decision_data_v1.npz](https://github.com/das1iw/decision-focused-chassis-world-models/releases/download/v1.0.0/decision_data_v1.npz) | 574,493,236 | `data/decision_data_v1.npz` |
| [external_baseline_step_logs_v1.zip](https://github.com/das1iw/decision-focused-chassis-world-models/releases/download/v1.0.0/external_baseline_step_logs_v1.zip) | 13,986,280 | Extract into `data/external_baseline_step_logs_v1/` |

Sequence data preserve recorded signals, commands, memory and course labels. Decision data preserve common candidate commands, twin values, windows, state/history and split identifiers. The external-baseline archive contains 216 saved numerical/metadata files. No new trajectories were collected for this release.

Verify files before use:

```text
python release_tools/verify_release.py --assets ../release_assets --out outputs/byte_verification.json
```

## Software and licensed dependencies

Use Python 3.10 or later. `requirements-core.txt` names the core packages; `manifests/current_host_smoke.json` records the packaging host's exact versions after the CPU smoke. This is a current-host environment record, not an assertion that the historical training environment was recovered. GPU planning additionally needs a compatible CUDA/CuPy setup; the source retains the original kernels. A clean second-machine and GPU reproduction remains to be completed.

The planar physics model loads tire, aligning-moment and motor tables from a licensed CarSim vehicle parameter asset. CarSim binaries and vehicle parameter files are not distributed here. `manifests/licensed_dependencies.json` identifies the expected expanded parameter asset by SHA-256. Offline model reproduction requires permitted access to that asset, even when the native solver is not run. Native closed-loop replication additionally requires CarSim 2024.1 and adaptation of the original host/solver paths.

Install core packages in an isolated environment and run the CPU smoke with the path to your own licensed asset:

```text
python -m pip install -r requirements-core.txt
python release_tools/smoke.py --asset-par PATH_TO_LICENSED_BASE_RUN_PAR --out outputs/smoke.json
```

The smoke initializes the physical model and strictly loads checkpoint state dictionaries. It does not execute a forecast, scientific score, training loop or simulator.

## Offline reproduction entry points

The following commands are provided for reproducing used-development analyses in a new output directory. They were not run as part of packaging. The preserved evaluation scripts import CuPy for checkpoint/kernel helpers even when executing on CPU, so install a compatible CuPy build before using them. The basic import/checkpoint smoke above does not require CuPy. The offline launcher must configure the parameter asset before the original modules capture its path.

```text
python release_tools/offline.py --asset-par PATH_TO_LICENSED_BASE_RUN_PAR tools/cwm289_resid_eval_v1.py --data data/sequence_data_v1.npz --mlp checkpoints/wm4_289/resid_dfl_v5/mlp_from_v3.pt --stride 20 --out outputs/forecast_reproduction.json
python release_tools/offline.py --asset-par PATH_TO_LICENSED_BASE_RUN_PAR tools/cwm289_dfl_score_offline_v1.py --dfl data/decision_data_v1.npz --models physics=physics,selected=resid:checkpoints/wm4_289/resid_dfl_v5/mlp_from_v3.pt --device cpu --out outputs/decision_reproduction.json
```

These evaluate used development data; they do not create fresh confirmation. Preserve original receipts and use separate output paths. Whole-course splits differ between forecast training (last four courses) and decision training (last three); inspect `specification/training_configs.json` before retraining. The per-decision normalization comparison has known batch/budget differences and is an empirical recipe.

Fresh-course tests are already used. Inspect the preserved confirmation/checker receipts for published statistics. The external-baseline step logs accompany them for audit. Packaging did not rerun frozen scorers or launch native campaigns. Native sources retain historical machine-specific paths and are provided for inspection; portable native replay is not claimed.

## Interpretation

The contribution is decision-focused force-residual training within explicit chassis dynamics and a fixed MPPI planner. The 30% result concerns an adapted neural-ExpTanh implementation under this vehicle, data and planner; it does not rank the original published controller in general. Forecast degradation, force-feasibility limits, simulation-provided initialization/prior information and lack of a strict sensing-to-actuation timing guarantee remain part of the result.

Code is released under MIT; datasets, numerical evidence and trained checkpoints under CC BY 4.0. See [LICENSE_STATUS.md](LICENSE_STATUS.md) for scope. Cite this dataset through the repository citation button or [CITATION.cff](CITATION.cff). The associated manuscript is named above; no paper DOI or journal acceptance is implied.
