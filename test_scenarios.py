"""
test_scenarios.py — Automated evaluation battery for the G1 stabilizer policy.

Runs 5 scenarios and prints a results table. Optionally saves videos.

Usage:
  python test_scenarios.py --model models/best/best_model
  python test_scenarios.py --model models/best/best_model --vecnorm models/vecnorm_final.pkl --save-video

Scenarios
---------
  1. Endurance        — stand still, no pushes, 30 s
  2. Cardinal pushes  — single push from front / back / left / right at 0.3 m/s
  3. Strong pushes    — 0.6 m/s push from a random direction every 3 s
  4. Rapid fire       — 0.4 m/s push every 1 s (stress test)
  5. Domain rand      — mass / friction / damping randomised, light pushes
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mediapy as media
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import EnvConfig
from envs.g1_stand_env import G1StandEnv


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    description: str
    episodes: int
    push_vel_max: float
    push_interval_steps: int     # control steps between pushes (50 Hz)
    push_directions: Optional[list]  # None = random; or list of (vx, vy) unit vecs
    use_domain_rand: bool
    duration_steps: int          # max steps per episode (50 Hz)


SCENARIOS = [
    Scenario(
        name="1. Endurance",
        description="Stand still, no disturbances, 30 s",
        episodes=3,
        push_vel_max=0.0,
        push_interval_steps=99999,
        push_directions=None,
        use_domain_rand=False,
        duration_steps=1500,     # 30 s at 50 Hz
    ),
    Scenario(
        name="2. Cardinal push",
        description="Single moderate push (0.3 m/s) from each of 4 directions",
        episodes=4,              # one per direction
        push_vel_max=0.3,
        push_interval_steps=99999,  # only one push (injected manually)
        push_directions=[
            (0.3,  0.0),   # forward
            (-0.3, 0.0),   # backward
            (0.0,  0.3),   # left
            (0.0, -0.3),   # right
        ],
        use_domain_rand=False,
        duration_steps=500,      # 10 s to recover
    ),
    Scenario(
        name="3. Strong pushes",
        description="0.6 m/s push every 3 s from random direction",
        episodes=5,
        push_vel_max=0.6,
        push_interval_steps=150,
        push_directions=None,
        use_domain_rand=False,
        duration_steps=1000,
    ),
    Scenario(
        name="4. Rapid fire",
        description="0.4 m/s push every 1 s (stress test)",
        episodes=5,
        push_vel_max=0.4,
        push_interval_steps=50,
        push_directions=None,
        use_domain_rand=False,
        duration_steps=1000,
    ),
    Scenario(
        name="5. Domain rand",
        description="Randomised mass / friction / damping + light pushes",
        episodes=5,
        push_vel_max=0.3,
        push_interval_steps=120,
        push_directions=None,
        use_domain_rand=True,
        duration_steps=1000,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_path: str, vec_env, vecnorm_path: Optional[str]):
    if vecnorm_path and Path(vecnorm_path).exists():
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device="cpu")
    return model, vec_env


def _make_env(cfg: EnvConfig):
    env = G1StandEnv(render_mode="rgb_array", config=cfg)
    return DummyVecEnv([lambda: env]), env


def _run_episode(
    model,
    vec_env,
    raw_env: G1StandEnv,
    scenario: Scenario,
    episode_idx: int,
    save_video: bool,
):
    obs = vec_env.reset()
    frames = []
    total_reward = 0.0
    survived_steps = 0
    done = False

    # For cardinal pushes: inject a single push at step 50 (1 s in)
    cardinal_push_done = False
    push_step = 50
    direction = None
    if scenario.push_directions:
        direction = scenario.push_directions[
            episode_idx % len(scenario.push_directions)
        ]

    while not done and survived_steps < scenario.duration_steps:
        # Manual cardinal push injection
        if direction and not cardinal_push_done and survived_steps == push_step:
            raw_env.data.qvel[3] += direction[0]
            raw_env.data.qvel[4] += direction[1]
            cardinal_push_done = True

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += float(reward[0])
        survived_steps += 1

        if save_video:
            frame = raw_env.render()
            if frame is not None:
                frames.append(frame)

    survived_frac = survived_steps / scenario.duration_steps
    return total_reward, survived_steps, survived_frac, frames


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all(model_path: str, vecnorm_path: Optional[str], save_video: bool):
    print("=" * 68)
    print("  G1 Stabilizer — Test Battery")
    print("=" * 68)

    results_summary = []

    for scenario in SCENARIOS:
        print(f"\n{scenario.name}")
        print(f"  {scenario.description}")
        print(f"  {'-' * 50}")

        cfg = EnvConfig()
        cfg.push_vel_max = scenario.push_vel_max
        cfg.push_interval_steps = scenario.push_interval_steps
        cfg.use_domain_rand = scenario.use_domain_rand
        cfg.max_episode_steps = scenario.duration_steps

        vec_env, raw_env = _make_env(cfg)
        model, vec_env = _load_model(model_path, vec_env, vecnorm_path)

        ep_rewards = []
        ep_lengths = []
        ep_surv = []

        for ep in range(scenario.episodes):
            reward, steps, frac, frames = _run_episode(
                model, vec_env, raw_env, scenario, ep, save_video
            )
            ep_rewards.append(reward)
            ep_lengths.append(steps)
            ep_surv.append(frac)

            dir_label = ""
            if scenario.push_directions:
                dirs = ["forward", "backward", "left", "right"]
                dir_label = f"  ({dirs[ep % len(dirs)]})"

            print(
                f"  ep {ep + 1:2d}{dir_label:<12}"
                f"  reward={reward:8.1f}"
                f"  steps={steps:5d}/{scenario.duration_steps}"
                f"  survived={frac * 100:5.1f}%"
            )

            if save_video and frames:
                out = f"test_s{scenario.name[0]}_ep{ep + 1}.mp4"
                media.write_video(out, np.stack(frames), fps=50)
                print(f"             → saved {out}")

        mean_surv = np.mean(ep_surv) * 100
        mean_rew = np.mean(ep_rewards)
        print(
            f"  ── mean survival: {mean_surv:.1f}%    mean reward: {mean_rew:.1f}"
        )
        results_summary.append((scenario.name, mean_surv, mean_rew))

        vec_env.close()

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print(f"  {'Scenario':<30}  {'Survival %':>10}  {'Mean Reward':>12}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
    for name, surv, rew in results_summary:
        print(f"  {name:<30}  {surv:>9.1f}%  {rew:>12.1f}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, metavar="PATH")
    parser.add_argument("--vecnorm", default=None, metavar="PATH")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    run_all(args.model, args.vecnorm, args.save_video)
