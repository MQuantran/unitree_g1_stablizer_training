# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics stabilizer training workspace targeting the **Unitree G1 humanoid robot** simulated in **MuJoCo**. The current repo contains reference documentation; simulation notebooks and training code are expected to live alongside these files.

## Key Reference Files

- **`REFERENCE.md`** — Quick-reference for five MuJoCo tutorial notebooks: `tutorial.ipynb`, `mjspec.ipynb`, `LQR.ipynb`, `rollout.ipynb`, `least_squares.ipynb`. Contains copy-paste-ready API snippets for the full simulation stack.
- **`unitree_manual_list_docs.txt`** — Full crawl of the Unitree G1 developer documentation (SDK architecture, DDS service interfaces, RL control routine, joint motor specs).

## Training Code Structure

```
stablizer_training/
├── config.py            # EnvConfig + TrainConfig dataclasses (all hyperparams)
├── train.py             # PPO training entry point
├── eval.py              # Load checkpoint, render episodes, save video
├── check_model.py       # Inspect loaded MJCF — validate joint/actuator names
├── setup.sh             # One-shot setup: clone menagerie, create venv, pip install
├── requirements.txt
└── envs/
    └── g1_stand_env.py  # Gymnasium env (G1StandEnv)
```

**Robot model** (not in repo — fetched by `setup.sh`):
```
mujoco_menagerie/unitree_g1/scene.xml   ← path set in EnvConfig.model_path
```

## Commands

```bash
# First-time setup on the training machine
bash setup.sh

# Verify joint/actuator names resolve correctly before training
python check_model.py

# Train from scratch
python train.py

# Resume from a checkpoint
python train.py --resume models/g1_stand_500000_steps

# Monitor training
tensorboard --logdir logs

# Evaluate a checkpoint (renders inline or saves .mp4)
python eval.py --model models/best/best_model --vecnorm models/vecnorm_final.pkl
python eval.py --model models/best/best_model --push 0.5 --save-video
```

## Core Stack

| Layer | Library |
|-------|---------|
| RL algorithm | `stable-baselines3` PPO |
| Physics | `mujoco` ≥ 3.1 |
| Parallelism | `SubprocVecEnv` (16 envs, `fork` on Linux) |
| Obs/reward norm | `VecNormalize` |
| Rendering / video | `mediapy` |

## Architecture Patterns

### Environment (`G1StandEnv`)
- **Control**: residual position control — policy outputs `Δq ∈ [-1,1]^13`, applied as `ctrl = default_qpos + action_scale × Δq`. The model's PD servos convert ctrl → torques internally.
- **Controlled joints**: 12 leg DoFs + 1 waist (arms held at default). Names in `G1StandEnv.ACT_JOINT_NAMES`.
- **Termination**: torso tilt > 40° or pelvis height < 0.4 m.
- **Curriculum hook**: `env.update_config(**kwargs)` is called via `vec_env.env_method()` by `CurriculumCallback` — this is how push magnitude and domain-rand are enabled mid-training without restarting.

### Curriculum stages (configured in `TrainConfig.curriculum_thresholds`)
| Stage | Trigger | Change |
|-------|---------|--------|
| 1 | start | no pushes, no domain-rand |
| 2 | 2 M steps | push ≤ 0.2 m/s |
| 3 | 10 M steps | push ≤ 0.5 m/s |
| 4 | 25 M steps | domain randomisation on |

### Domain randomisation (episode reset, `EnvConfig.use_domain_rand = True`)
Randomises body mass (±15%), ground friction (±30%), joint damping (±30%) each episode reset. Nominal values are saved once at first reset.

### If joint names don't resolve
Run `python check_model.py` — it prints every joint, actuator, and body in the loaded model and flags which `ACT_JOINT_NAMES` entries are missing. Update `G1StandEnv.ACT_JOINT_NAMES`, `TORSO_BODY`, `LEFT_FOOT_BODY`, or `RIGHT_FOOT_BODY` to match.

### Procedural model editing (mjSpec) — see `REFERENCE.md §7`
Use `mujoco.MjSpec` to add heightfields, meshes, or bodies programmatically. Call `spec.compile()` after every edit.

## Unitree G1 Specifics

- **DOF**: 23 (basic) to 43 (EDU) — 6 per leg, 5 per arm, 1 waist. Menagerie model has 37.
- **SDK communication**: DDS-based service interfaces (Robot State, Sport, Odometer, VuiClient, SLAM/Nav, Motion Switcher) — documented in `unitree_manual_list_docs.txt`.
- **RL integration entry point**: `unitree_sdk2` RL Control Routine (same doc, "RL Control Routine" section).
- Joint indices follow the ordering in `unitree_manual_list_docs.txt` (Joint Motor Sequence section).
