# Data guide

## 1. Recorded trajectories

`sequence_data_v1.npz` contains driving sequences from the 44-state chassis twin. Two groups, `m90` and `m75`, use speed margins 0.90 and 0.75. Each has 128 episodes with up to 2,000 samples at 0.02 s intervals. Use the `live` mask to exclude samples after an episode ends.

| Field suffix | Meaning |
|---|---|
| `sig` | Longitudinal/lateral velocity, yaw rate, four wheel speeds, four realized motor torques, longitudinal/lateral acceleration. |
| `pose` | World position x/y and heading. |
| `u` | Four steering commands and four motor-torque commands. |
| `h` | The reduced model's 18 physical-memory variables. |
| `mu`, `mu_in` | True simulated friction and the planner's friction input. |
| `course`, `mu0`, `t_drop` | Course ID, initial friction and grip-loss time. |
| `live` | Valid episode samples. |

The runs cover 16 courses, four friction levels, nominal driving and sudden grip loss. Forecast training uses 192 episodes; the 64 episodes on courses 9039–9042 are held out.

## 2. Planning decisions

`decision_data_v1.npz` stores what the twin predicts for alternative commands from a common starting situation.

| Field | Shape | Meaning |
|---|---|---|
| `A` | 5120 × 64 × 75 × 6 | Candidate front/rear steering and four wheel-torque commands. |
| `Gtw` | 5120 × 64 | Twin planning values; larger is preferred. |
| `alive_tw` | 5120 × 64 | Whether each twin rollout survives the prediction horizon. |
| `s0`, `h0`, `pose0` | Per situation | Mechanical state, physical memory and pose at the start. |
| `st0`, `ts0`, `prev_a` | Per situation | Steering/torque limiter states and previous command. |
| `txy`, `tpsi`, `tsw`, `bref`, `vl` | Per situation | Course geometry, sideslip target and speed envelope. |
| `hist_*` | Ten steps per situation | State, memory, command and friction history. |
| `course`, `tb`, `k` | Per situation | Original course and sampling identifiers. |

Each candidate spans 1.5 s. The bank keeps the twin's 32 best candidates and 32 random others. Decision training uses 4,160 situations; the 960 situations on courses 9040–9042 are held out. The stored sequences are hypothetical candidate commands, not independent executed drives.

## 3. CarSim validation

`external_baseline_step_logs_v1.zip` contains 48 drives for each of four arms: physics, adapted ExpTanh, and two independently trained decision-focused residuals. The primary comparison uses 16 courses at friction 0.75/0.95; eight of those courses also run at 0.35/0.55.

Extract the archive into `data/carsim/`. These are the logs of the adapted learned-tire comparison; the earlier CarSim confirmations are not included as step logs.

Step-log columns are time, course progress, actual sideslip, reference sideslip, lateral error and the planner's friction estimate. JSON/JSONL files contain drive conditions and summaries. The archive retains original arm/course filenames so the numerical records remain traceable.

## 4. Units and use

Mechanical quantities use SI units. Wheel order is front left, front right, rear left, rear right. Sequence steering commands are radians and wheel torques are N m. Candidate commands in `A` are normalized to [-1, 1].

Split by complete courses when evaluating generalization. These data have already been used in the study; reusing them does not create a new prospective test. File sizes and SHA-256 hashes are listed in `metadata/data.json` and `SHA256SUMS.txt`. To check the downloads, keep all three archives and `SHA256SUMS.txt` in one directory and run `sha256sum -c SHA256SUMS.txt` there before extracting the CarSim archive.
