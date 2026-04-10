"""
interactive.py — Live MuJoCo viewer with keyboard control.

The RL stabilizer policy runs continuously. You layer commands on top:
  - Push the robot to test balance recovery
  - Wave arms with sinusoidal overlays
  - Reset the episode

Requires a display (run on the training machine directly, or via X11 forwarding).

Usage:
  python interactive.py --model models/best/best_model
  python interactive.py --model models/best/best_model --vecnorm models/vecnorm_final.pkl

Controls
--------
  W / S          Push robot forward / backward  (0.4 m/s)
  A / D          Push robot left / right        (0.4 m/s)
  SHIFT + W/S/A/D  Strong push                  (0.8 m/s)
  SPACE          Random push from any direction
  1              Wave left arm
  2              Wave right arm
  3              Wave both arms
  4              Stop arm motion
  R              Reset episode
  ESC / Q        Quit

NOTE: "Move forward" is not supported by this stabilizer policy — it was
trained only to stand still. The keyboard pushes let you test push recovery.
For locomotion (walking forward), a separate velocity-conditioned policy
would be needed and trained on top of this stabilizer.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import EnvConfig
from envs.g1_stand_env import G1StandEnv


# ─────────────────────────────────────────────────────────────────────────────
# Arm-wave motion generator
# ─────────────────────────────────────────────────────────────────────────────

# Joint names for left / right arms in the menagerie G1 model.
# Adjust if your model uses different names (check_model.py will show them).
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# Wave amplitudes (radians) and offsets per joint for a natural-looking wave
LEFT_WAVE_AMP    = [0.6, 0.3, 0.2, 0.5]
LEFT_WAVE_OFFSET = [0.4, 0.3, 0.0, 0.8]
RIGHT_WAVE_AMP    = [0.6, -0.3, -0.2, 0.5]
RIGHT_WAVE_OFFSET = [0.4, -0.3, 0.0, 0.8]
WAVE_FREQ = 1.5   # Hz


class ArmWave:
    """
    Generates sinusoidal joint targets for arm waving.
    Applied on top of the standing policy (which controls only legs).
    """

    def __init__(self, model: mujoco.MjModel):
        self._model = model
        self._left_ids = self._resolve(LEFT_ARM_JOINTS)
        self._right_ids = self._resolve(RIGHT_ARM_JOINTS)
        self.mode = 0   # 0 = off, 1 = left, 2 = right, 3 = both

    def _resolve(self, names: list[str]) -> list[int]:
        ids = []
        for name in names:
            act_name = name.replace("_joint", "")
            idx = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )
            if idx == -1:
                idx = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
            ids.append(idx)  # -1 means not found; we skip those below
        return ids

    def apply(self, data: mujoco.MjData, t: float):
        """Write arm ctrl targets into data.ctrl for the current sim time t."""
        if self.mode == 0:
            return

        phase = 2 * np.pi * WAVE_FREQ * t

        if self.mode in (1, 3):
            for i, act_id in enumerate(self._left_ids):
                if act_id == -1:
                    continue
                target = LEFT_WAVE_OFFSET[i] + LEFT_WAVE_AMP[i] * np.sin(phase + i)
                lo, hi = self._model.actuator_ctrlrange[act_id]
                data.ctrl[act_id] = float(np.clip(target, lo, hi))

        if self.mode in (2, 3):
            for i, act_id in enumerate(self._right_ids):
                if act_id == -1:
                    continue
                target = RIGHT_WAVE_OFFSET[i] + RIGHT_WAVE_AMP[i] * np.sin(phase + i)
                lo, hi = self._model.actuator_ctrlrange[act_id]
                data.ctrl[act_id] = float(np.clip(target, lo, hi))


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard listener (runs in a background thread via pynput)
# ─────────────────────────────────────────────────────────────────────────────

class KeyboardController:
    def __init__(self):
        self._pressed: set = set()
        self._events: list = []   # one-shot events queued for the sim thread
        self._lock = threading.Lock()
        self._running = True

    def start(self):
        from pynput import keyboard

        def on_press(key):
            try:
                self._pressed.add(key.char.lower())
            except AttributeError:
                self._pressed.add(key)

        def on_release(key):
            try:
                self._pressed.discard(key.char.lower())
            except AttributeError:
                self._pressed.discard(key)
            # One-shot commands on release
            try:
                ch = key.char.lower()
            except AttributeError:
                ch = None
            with self._lock:
                if ch in ("1", "2", "3", "4", "r"):
                    self._events.append(ch)
                if key == keyboard.Key.space:
                    self._events.append("space")
                if key in (keyboard.Key.esc,) or ch == "q":
                    self._events.append("quit")

        self._listener = keyboard.Listener(
            on_press=on_press, on_release=on_release
        )
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def pop_events(self) -> list:
        with self._lock:
            evts = list(self._events)
            self._events.clear()
        return evts

    def is_pressed(self, char: str) -> bool:
        return char in self._pressed

    def shift_held(self) -> bool:
        from pynput import keyboard
        return (
            keyboard.Key.shift in self._pressed
            or keyboard.Key.shift_l in self._pressed
            or keyboard.Key.shift_r in self._pressed
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main interactive loop
# ─────────────────────────────────────────────────────────────────────────────

def run(model_path: str, vecnorm_path: str | None):
    cfg = EnvConfig()
    cfg.push_vel_max = 0.0     # pushes handled manually from keyboard
    cfg.max_episode_steps = 999999

    raw_env = G1StandEnv(render_mode="rgb_array", config=cfg)
    vec_env = DummyVecEnv([lambda: raw_env])

    if vecnorm_path and Path(vecnorm_path).exists():
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    policy = PPO.load(model_path, env=vec_env, device="cpu")
    arm_wave = ArmWave(raw_env.model)
    kb = KeyboardController()
    kb.start()

    print("\nG1 Interactive Controller")
    print("─" * 40)
    print("  W/S/A/D          Push (0.4 m/s)")
    print("  SHIFT+W/S/A/D    Strong push (0.8 m/s)")
    print("  SPACE            Random push")
    print("  1  Wave left arm")
    print("  2  Wave right arm")
    print("  3  Wave both arms")
    print("  4  Stop arm wave")
    print("  R  Reset episode")
    print("  Q / ESC  Quit")
    print("─" * 40)
    print("NOTE: This policy was trained to stand still.")
    print("      Pushes test balance recovery, not locomotion.\n")

    obs = vec_env.reset()
    quit_flag = False

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135

        while viewer.is_running() and not quit_flag:
            # ── One-shot events ───────────────────────────────────────────────
            for evt in kb.pop_events():
                if evt == "quit":
                    quit_flag = True

                elif evt == "r":
                    obs = vec_env.reset()
                    print("[Reset] Episode restarted.")

                elif evt == "space":
                    angle = np.random.uniform(0, 2 * np.pi)
                    mag = np.random.uniform(0.3, 0.6)
                    raw_env.data.qvel[3] += mag * np.cos(angle)
                    raw_env.data.qvel[4] += mag * np.sin(angle)
                    print(f"[Push] random  {mag:.2f} m/s")

                elif evt == "1":
                    arm_wave.mode = 1
                    print("[Arm] Left arm waving")
                elif evt == "2":
                    arm_wave.mode = 2
                    print("[Arm] Right arm waving")
                elif evt == "3":
                    arm_wave.mode = 3
                    print("[Arm] Both arms waving")
                elif evt == "4":
                    arm_wave.mode = 0
                    print("[Arm] Arm motion stopped")

            # ── Held-key pushes ────────────────────────────────────────────────
            push_mag = 0.8 if kb.shift_held() else 0.4
            pushed = False
            if kb.is_pressed("w"):
                raw_env.data.qvel[3] += push_mag * 0.05
                pushed = True
            if kb.is_pressed("s"):
                raw_env.data.qvel[3] -= push_mag * 0.05
                pushed = True
            if kb.is_pressed("a"):
                raw_env.data.qvel[4] += push_mag * 0.05
                pushed = True
            if kb.is_pressed("d"):
                raw_env.data.qvel[4] -= push_mag * 0.05
                pushed = True

            # ── Policy step ────────────────────────────────────────────────────
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)

            # ── Arm wave overlay ────────────────────────────────────────────────
            arm_wave.apply(raw_env.data, raw_env.data.time)

            # ── Auto-reset on fall ────────────────────────────────────────────
            if done:
                print("[Episode] Robot fell — auto-resetting.")
                obs = vec_env.reset()

            viewer.sync()

    kb.stop()
    vec_env.close()
    print("Exited.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, metavar="PATH")
    parser.add_argument("--vecnorm", default=None, metavar="PATH")
    args = parser.parse_args()

    run(args.model, args.vecnorm)
