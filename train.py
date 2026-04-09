"""
train.py — PPO training for G1 standing balance.

Usage:
  python train.py                        # fresh run
  python train.py --resume models/g1_stand_500000_steps  # continue from checkpoint

Tensorboard:
  tensorboard --logdir logs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from config import EnvConfig, TrainConfig
from envs.g1_stand_env import G1StandEnv


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_fn(cfg: EnvConfig, rank: int, seed: int = 0):
    """Returns a callable that constructs one monitored G1StandEnv."""
    import copy

    def _init():
        env_cfg = copy.deepcopy(cfg)
        env = G1StandEnv(config=env_cfg)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum callback
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Advances training through 4 stages based on total env-step count.

    Stage 1  (0 steps)       — no disturbances, no domain-rand
    Stage 2  (2 M steps)     — small pushes (0.2 m/s)
    Stage 3  (10 M steps)    — larger pushes (0.5 m/s)
    Stage 4  (25 M steps)    — full domain randomisation
    """

    def __init__(self, train_cfg: TrainConfig, verbose: int = 1):
        super().__init__(verbose)
        self._thresholds = train_cfg.curriculum_thresholds
        self._stage = 1

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        new_stage = 1 + sum(steps >= t for t in self._thresholds)

        if new_stage > self._stage:
            self._stage = new_stage
            self._advance(new_stage)

        return True  # continue training

    def _advance(self, stage: int):
        if self.verbose:
            print(f"\n[Curriculum] ── Stage {stage} ──────────────────────────")

        if stage == 2:
            self.training_env.env_method(
                "update_config", push_vel_max=0.2, push_interval_steps=150
            )
            if self.verbose:
                print("  push_vel_max=0.2 m/s, interval=150 steps")

        elif stage == 3:
            self.training_env.env_method(
                "update_config", push_vel_max=0.5, push_interval_steps=100
            )
            if self.verbose:
                print("  push_vel_max=0.5 m/s, interval=100 steps")

        elif stage == 4:
            self.training_env.env_method("update_config", use_domain_rand=True)
            if self.verbose:
                print("  domain randomisation enabled")


# ─────────────────────────────────────────────────────────────────────────────
# VecNormalize sync (keep eval stats current with training stats)
# ─────────────────────────────────────────────────────────────────────────────

class SyncVecNormCallback(BaseCallback):
    """Copy running obs stats from the training VecNormalize to the eval one."""

    def __init__(self, eval_vec_norm: VecNormalize, sync_freq: int = 1):
        super().__init__()
        self._eval_vn = eval_vec_norm
        self._sync_freq = sync_freq
        self._calls = 0

    def _on_step(self) -> bool:
        self._calls += 1
        if self._calls % self._sync_freq == 0:
            # self.model.get_env() is the training VecNormalize
            train_vn = self.model.get_env()
            self._eval_vn.obs_rms = train_vn.obs_rms
            self._eval_vn.ret_rms = train_vn.ret_rms
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(resume: str | None = None):
    env_cfg = TrainConfig()      # naming note: env_cfg ↔ TrainConfig
    train_cfg = TrainConfig()
    env_base_cfg = EnvConfig()   # stage 1: no pushes, no rand

    # Make output directories
    Path(train_cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(train_cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # ── Training envs ─────────────────────────────────────────────────────────
    print(f"Spawning {train_cfg.n_envs} training environments…")
    vec_env = SubprocVecEnv(
        [_make_env_fn(env_base_cfg, rank=i) for i in range(train_cfg.n_envs)],
        start_method="fork",   # 'fork' on Linux; change to 'spawn' on Windows
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Eval env (stats updated from training env, rewards not normalised) ────
    eval_env_cfg = EnvConfig()
    eval_env_cfg.push_vel_max = 0.3     # fixed moderate push during eval
    eval_env_cfg.push_interval_steps = 100
    eval_vec = SubprocVecEnv(
        [_make_env_fn(eval_env_cfg, rank=999)],
        start_method="fork",
    )
    eval_vec_norm = VecNormalize(
        eval_vec, norm_obs=True, norm_reward=False, training=False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=train_cfg.net_arch, vf=train_cfg.net_arch),
        activation_fn=__import__("torch.nn", fromlist=["Tanh"]).Tanh,
    )

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(
            resume,
            env=vec_env,
            device="cuda",
            tensorboard_log=train_cfg.log_dir,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=train_cfg.learning_rate,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            ent_coef=train_cfg.ent_coef,
            vf_coef=train_cfg.vf_coef,
            max_grad_norm=train_cfg.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=train_cfg.log_dir,
            verbose=1,
            device="cuda",
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.save_freq // train_cfg.n_envs,
        save_path=train_cfg.save_dir,
        name_prefix="g1_stand",
        save_vecnormalize=True,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_vec_norm,
        best_model_save_path=f"{train_cfg.save_dir}/best",
        log_path=train_cfg.log_dir,
        eval_freq=train_cfg.eval_freq // train_cfg.n_envs,
        n_eval_episodes=train_cfg.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    curriculum_cb = CurriculumCallback(train_cfg, verbose=1)
    sync_vn_cb = SyncVecNormCallback(eval_vec_norm, sync_freq=256)

    callbacks = CallbackList([checkpoint_cb, eval_cb, curriculum_cb, sync_vn_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting PPO training")
    print(f"  Total steps : {train_cfg.total_timesteps:,}")
    print(f"  Envs        : {train_cfg.n_envs}")
    print(f"  Device      : cuda (RTX 3060)")
    print(f"  Logs        : {train_cfg.log_dir}/")
    print(f"  Models      : {train_cfg.save_dir}/\n")

    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
        progress_bar=True,
    )

    # ── Save final ────────────────────────────────────────────────────────────
    model.save(f"{train_cfg.save_dir}/g1_stand_final")
    vec_env.save(f"{train_cfg.save_dir}/vecnorm_final.pkl")
    print("Done — final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a .zip checkpoint to resume from",
    )
    args = parser.parse_args()
    train(resume=args.resume)
