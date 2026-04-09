"""
eval.py — Load a trained G1 standing policy and render it.

Usage:
  # Watch the best saved model (5 episodes, renders video in notebook/display)
  python eval.py --model models/best/best_model

  # Pass a specific VecNormalize stats file
  python eval.py --model models/g1_stand_final --vecnorm models/vecnorm_final.pkl

  # Save video frames to disk instead of showing them
  python eval.py --model models/best/best_model --save-video

  # Test robustness: enable strong pushes
  python eval.py --model models/best/best_model --push 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import EnvConfig
from envs.g1_stand_env import G1StandEnv


def make_eval_env(cfg: EnvConfig):
    env = G1StandEnv(render_mode="rgb_array", config=cfg)
    return env


def evaluate(
    model_path: str,
    vecnorm_path: str | None = None,
    n_episodes: int = 5,
    push_vel: float = 0.0,
    save_video: bool = False,
):
    cfg = EnvConfig()
    cfg.push_vel_max = push_vel
    cfg.push_interval_steps = 80 if push_vel > 0 else 99999
    cfg.use_domain_rand = False   # deterministic eval

    raw_env = make_eval_env(cfg)
    vec_env = DummyVecEnv([lambda: raw_env])

    if vecnorm_path and Path(vecnorm_path).exists():
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        if vecnorm_path:
            print(f"Warning: {vecnorm_path} not found — running without normalisation.")

    model = PPO.load(model_path, env=vec_env, device="cpu")
    print(f"Model loaded: {model_path}")
    print(f"Obs dim: {vec_env.observation_space.shape[0]}")
    print(f"Act dim: {vec_env.action_space.shape[0]}")
    print()

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        frames = []
        ep_reward = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1

            frame = raw_env.render()
            if frame is not None:
                frames.append(frame)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        print(f"Episode {ep + 1:2d} | reward={ep_reward:8.1f} | steps={ep_len:4d}")

        if frames:
            fps = int(1.0 / (cfg.sim_timestep * cfg.control_decimation))
            if save_video:
                out_path = f"eval_ep{ep + 1}.mp4"
                media.write_video(out_path, np.stack(frames), fps=fps)
                print(f"           Video saved → {out_path}")
            else:
                media.show_video(np.stack(frames), fps=fps)

    print()
    print(f"Mean reward : {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Mean length : {np.mean(episode_lengths):.0f} steps  "
          f"({np.mean(episode_lengths) * cfg.sim_timestep * cfg.control_decimation:.1f} s)")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, metavar="PATH",
        help="Path to saved PPO model (.zip, pass without extension)",
    )
    parser.add_argument(
        "--vecnorm", default=None, metavar="PATH",
        help="Path to VecNormalize stats file (.pkl)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--push", type=float, default=0.0, metavar="M/S",
        help="Max push velocity to apply during eval in m/s (default: 0 = off)",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Write .mp4 files instead of showing inline",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.episodes,
        push_vel=args.push,
        save_video=args.save_video,
    )
