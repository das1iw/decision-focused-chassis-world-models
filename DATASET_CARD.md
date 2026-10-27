# Dataset card

## Associated paper

**Physics-Structured Chassis World Models with Decision-Focused Tire-Force Learning**  
Dastan Kaldybayev, Yanzhao Su, Jin Huang and Guangyu Tian. Manuscript, 2026.

This is the dataset and companion-code repository for that manuscript. No journal acceptance or paper DOI is claimed. Stable versioned downloads: [v1.0.0](https://github.com/das1iw/decision-focused-chassis-world-models/releases/tag/v1.0.0). Cite the dataset using [CITATION.cff](CITATION.cff).

## Data components

1. **Recorded twin sequences** (`sequence_data_v1.npz`): 256 episodes over 16 course geometries (9027--9042), friction 0.35/0.55/0.75/0.95, nominal and 0.7-times grip-loss conditions, and speed margins 0.9/0.75. Arrays have 128 episodes per margin, up to 2000 slots per episode at 0.02 s. Use `live` masks; padded/post-terminal slots are not extra observations. Signals contain body velocities, yaw rate, four wheel speeds, four realized motor torques and accelerations. Commands, pose, observer memory, true/input friction, course and drop-time labels are also stored.
2. **Reference decision bank** (`decision_data_v1.npz`): 5120 situations, each with 64 candidate sequences of 75 steps and six normalized actions. `A` has shape (5120,64,75,6); `Gtw` and `alive_tw` give twin values and rollout-validity flags. Each bank comprises the twin's 32 best and 32 random other candidates. Initialization, physical memory, track windows, prior commands and ten-step history are retained. These are hypothetical command sequences evaluated by the twin, not 327680 independent executed drives.
3. **CarSim external-baseline records** (`external_baseline_step_logs_v1.zip`): four planning arms (physics, adapted ExpTanh and two decision-focused replicas), each with 48 nominal drives. Sixteen courses at friction 0.75/0.95 and eight of those at 0.35/0.55 give 192 drives total. Per-step arrays record time, station progress, actual/reference sideslip, lateral error and planner friction estimate; JSON/JSONL records provide cell metadata and summaries. These are the saved records supporting the manuscript's adapted-baseline comparison.

## Learning splits and interpretation

Forecast training excludes courses 9039--9042 (192 training and 64 validation episodes). Decision training excludes 9040--9042 (4160 training and 960 held-out decisions). Use course labels, not random sample-level splits. Data have already been used for development/confirmation; reuse is not fresh prospective validation. Course geometry is the statistical group for the reported closed-loop comparisons.

Motion/decision training data come from the 44-state simulation twin. The external validation logs come from CarSim. This is not real-vehicle data or a direct ground-truth tire-force dataset. Simulation-supplied state/memory and initial-friction priors are disclosed in the implementation specification. Source code and coefficients alone do not provide the licensed tire/motor tables required for full model/native replication.

## Format, units and dependencies

NPZ files contain ordinary NumPy arrays. Canonical wheel order is front left, front right, rear left, rear right. Mechanical variables use SI units; normalized candidate commands lie in [-1,1]. Sequence commands contain four steering angles in radians and four wheel motor torques in N m. Decision action channels are front/rear steering and four wheel torques. See source docstrings and `specification/implementation.tex` for exact fields and modeling settings.

Native CarSim software and proprietary vehicle parameter assets are excluded. Data browsing and reading receipts do not require the native solver. Re-evaluating the physical model requires permitted access to the parameter asset identified in `manifests/licensed_dependencies.json`.

## Integrity and licenses

`manifests/release_assets.json` lists sizes, SHA-256 hashes and download URLs. `manifests/source_files.json` records preserved source/result hashes. CPU checkpoint loading and byte checks are documented; a clean second-machine/GPU reproduction is not claimed.

Code: MIT. Data, numerical evidence and checkpoints: CC BY 4.0. See `LICENSE_STATUS.md` for scope. Attribution should identify the dataset version and companion paper.
