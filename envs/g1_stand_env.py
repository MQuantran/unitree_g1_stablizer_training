"""
G1StandEnv — Gymnasium environment for Unitree G1 upright-balance training.

Objective: keep the robot standing under random push disturbances and, later,
under full domain randomisation.

Control scheme: residual position control.
  action ∈ [-1, 1]^13  (normalised)
  ctrl   = default_standing_qpos + action_scale × action
  The model's internal PD servos convert ctrl to joint torques.

Observations (79-dim by default with 13 controlled joints):
  [4]  torso quaternion (w, x, y, z)
  [3]  torso angular velocity  (world frame)
  [3]  torso linear velocity   (world frame)
  [13] joint positions relative to default standing pose
  [13] joint velocities
  [2]  foot contact flags (left, right)  ∈ {0, 1}
  [13] previous action
  ────
  57 total  (scales with n_act_joints)

Reward:
  r_upright    — dot(torso_z_axis, world_z_axis);  1.0 when perfectly upright
  r_height     — Gaussian centred on target CoM height
  r_contact    — mean foot-contact indicator
  r_joint_vel  — negative squared joint velocities
  r_action_rate — negative squared action delta (smoothness)
  r_energy     — negative squared actuator forces
  r_survival   — constant bonus per step while alive
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class G1StandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # ── Joint names to control (lower body + waist = 13 DoFs) ───────────────
    # These match the mujoco_menagerie/unitree_g1 model.
    # Arms are left at their default positions (not actuated by the policy).
    ACT_JOINT_NAMES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
    ]

    # Body names used for reward / termination
    TORSO_BODY = "pelvis"
    LEFT_FOOT_BODY = "left_ankle_roll_link"
    RIGHT_FOOT_BODY = "right_ankle_roll_link"

    def __init__(self, render_mode: Optional[str] = None, config=None):
        super().__init__()

        from config import EnvConfig
        self.cfg = config if config is not None else EnvConfig()

        # ── Load model ────────────────────────────────────────────────────────
        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run setup.sh to clone mujoco_menagerie."
            )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.sim_timestep

        # ── Identify joints / actuators / bodies ─────────────────────────────
        self._setup_joints()

        # ── Default standing pose (keyframe 0) ───────────────────────────────
        self._load_default_pose()

        # ── Nominal physical parameters for domain-rand reset ─────────────────
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._nominal_dof_damping = self.model.dof_damping.copy()

        # ── Spaces ───────────────────────────────────────────────────────────
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # ── Episode state ────────────────────────────────────────────────────
        self._step_count: int = 0
        self._push_timer: int = 0
        self._prev_action = np.zeros(self.n_joints, dtype=np.float32)

        # ── Renderer (optional) ──────────────────────────────────────────────
        self.render_mode = render_mode
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, width=640, height=480)

    # ─────────────────────────── Setup helpers ───────────────────────────────

    def _setup_joints(self):
        """
        Resolve joint names → qpos/qvel address indices and actuator ctrl indices.
        Also find body IDs for torso and feet.
        """

        def jnt_id(name: str) -> int:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx == -1:
                raise ValueError(
                    f"Joint '{name}' not found in model.\n"
                    "If the model uses different joint names, update ACT_JOINT_NAMES."
                )
            return idx

        def act_id(joint_name: str) -> int:
            # Menagerie convention: actuator name = joint name without '_joint' suffix.
            for candidate in (
                joint_name.replace("_joint", ""),
                joint_name,
            ):
                idx = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate
                )
                if idx != -1:
                    return idx
            raise ValueError(
                f"Actuator for joint '{joint_name}' not found.\n"
                "Run python check_model.py to list available actuators."
            )

        self.n_joints = len(self.ACT_JOINT_NAMES)
        jnt_ids = [jnt_id(n) for n in self.ACT_JOINT_NAMES]
        self.qpos_ids = [self.model.jnt_qposadr[i] for i in jnt_ids]
        self.qvel_ids = [self.model.jnt_dofadr[i] for i in jnt_ids]
        self.act_ids = [act_id(n) for n in self.ACT_JOINT_NAMES]

        def body_id(name: str) -> int:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx == -1:
                raise ValueError(f"Body '{name}' not found in model.")
            return idx

        self.torso_id = body_id(self.TORSO_BODY)
        self.left_foot_id = body_id(self.LEFT_FOOT_BODY)
        self.right_foot_id = body_id(self.RIGHT_FOOT_BODY)

    def _load_default_pose(self):
        """Populate self.default_qpos and self.default_act_qpos from keyframe 0."""
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.default_qpos = self.data.qpos.copy()
        self.default_ctrl = self.data.ctrl.copy()
        self.default_act_qpos = np.array(
            [self.data.qpos[i] for i in self.qpos_ids], dtype=np.float64
        )

    # ─────────────────────────── Observation ────────────────────────────────

    def _obs_dim(self) -> int:
        return 4 + 3 + 3 + self.n_joints + self.n_joints + 2 + self.n_joints

    def _get_obs(self) -> np.ndarray:
        # Torso quaternion (w, x, y, z)
        torso_quat = self.data.xquat[self.torso_id].copy()

        # Torso velocity in world frame: cvel rows are [angvel(3), linvel(3)]
        torso_angvel = self.data.cvel[self.torso_id, :3].copy()
        torso_linvel = self.data.cvel[self.torso_id, 3:].copy()

        # Actuated joint positions relative to default standing pose
        act_qpos = np.array([self.data.qpos[i] for i in self.qpos_ids])
        rel_qpos = act_qpos - self.default_act_qpos

        # Actuated joint velocities
        act_qvel = np.array([self.data.qvel[i] for i in self.qvel_ids])

        # Foot contacts
        left_c = float(self._foot_contact(self.left_foot_id))
        right_c = float(self._foot_contact(self.right_foot_id))

        return np.concatenate([
            torso_quat,
            torso_angvel,
            torso_linvel,
            rel_qpos,
            act_qvel,
            [left_c, right_c],
            self._prev_action,
        ]).astype(np.float32)

    def _foot_contact(self, body_id: int) -> bool:
        """Return True if any geom belonging to body_id is in contact."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (self.model.geom_bodyid[c.geom1] == body_id or
                    self.model.geom_bodyid[c.geom2] == body_id):
                return True
        return False

    # ─────────────────────────── Reward ─────────────────────────────────────

    def _compute_reward(self, action: np.ndarray):
        cfg = self.cfg

        # Torso z-axis in world frame (up direction of the pelvis)
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, self.data.xquat[self.torso_id])
        torso_up = rot.reshape(3, 3)[:, 2]          # 3rd column = local z
        upright_val = float(np.dot(torso_up, [0.0, 0.0, 1.0]))

        # 1. Upright
        r_upright = cfg.w_upright * upright_val

        # 2. CoM height
        com_h = float(self.data.subtree_com[self.torso_id, 2])
        r_height = cfg.w_height * float(
            np.exp(-cfg.height_sigma * abs(com_h - cfg.target_height))
        )

        # 3. Foot contact
        lc = float(self._foot_contact(self.left_foot_id))
        rc = float(self._foot_contact(self.right_foot_id))
        r_contact = cfg.w_contact * (lc + rc) / 2.0

        # 4. Joint velocity penalty
        act_qvel = np.array([self.data.qvel[i] for i in self.qvel_ids])
        r_joint_vel = -cfg.w_joint_vel * float(np.sum(act_qvel ** 2))

        # 5. Action rate penalty (smoothness)
        r_action_rate = -cfg.w_action_rate * float(
            np.sum((action - self._prev_action) ** 2)
        )

        # 6. Energy / torque penalty
        torques = self.data.actuator_force[self.act_ids]
        r_energy = -cfg.w_energy * float(np.sum(torques ** 2))

        # 7. Survival bonus
        r_survival = cfg.w_survival

        total = (
            r_upright + r_height + r_contact
            + r_joint_vel + r_action_rate + r_energy + r_survival
        )

        info = {
            "r_upright": r_upright,
            "r_height": r_height,
            "r_contact": r_contact,
            "r_joint_vel": r_joint_vel,
            "r_action_rate": r_action_rate,
            "r_energy": r_energy,
            "upright_val": upright_val,
            "com_height": com_h,
        }
        return total, info

    def _is_terminated(self) -> bool:
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, self.data.xquat[self.torso_id])
        torso_up = rot.reshape(3, 3)[:, 2]
        upright_val = np.dot(torso_up, [0.0, 0.0, 1.0])

        com_h = float(self.data.subtree_com[self.torso_id, 2])
        cos_max = np.cos(np.deg2rad(self.cfg.max_tilt_deg))

        return bool(upright_val < cos_max or com_h < self.cfg.min_height)

    # ─────────────────────────── Disturbance ────────────────────────────────

    def _maybe_push(self):
        """Randomly kick root linear velocity every push_interval_steps."""
        if self.cfg.push_vel_max <= 0.0:
            return
        self._push_timer += 1
        if self._push_timer >= self.cfg.push_interval_steps:
            self._push_timer = 0
            # Random horizontal impulse
            push = np.random.uniform(
                -self.cfg.push_vel_max, self.cfg.push_vel_max, 2
            )
            # Root freejoint linear velocity lives at indices 3, 4 of qvel
            self.data.qvel[3] += push[0]
            self.data.qvel[4] += push[1]

    # ─────────────────────────── Domain randomisation ────────────────────────

    def _apply_domain_rand(self):
        if not self.cfg.use_domain_rand:
            return
        cfg = self.cfg

        # Body mass
        scale = np.random.uniform(
            1.0 - cfg.mass_rand_ratio,
            1.0 + cfg.mass_rand_ratio,
            self.model.nbody,
        )
        self.model.body_mass[:] = self._nominal_body_mass * scale

        # Ground friction (lateral component)
        scale = np.random.uniform(
            1.0 - cfg.friction_rand_ratio,
            1.0 + cfg.friction_rand_ratio,
            self.model.ngeom,
        )
        self.model.geom_friction[:, 0] = self._nominal_geom_friction[:, 0] * scale

        # Joint damping
        scale = np.random.uniform(
            1.0 - cfg.damping_rand_ratio,
            1.0 + cfg.damping_rand_ratio,
            self.model.nv,
        )
        self.model.dof_damping[:] = self._nominal_dof_damping * scale

    # ─────────────────────────── Gym API ────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._apply_domain_rand()

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Small random perturbation around the standing pose
        self.data.qpos[:] += np.random.uniform(
            -self.cfg.init_noise, self.cfg.init_noise, self.model.nq
        )
        # Randomise the initial push timer so not all envs push at the same time
        self._push_timer = np.random.randint(
            0, max(1, self.cfg.push_interval_steps)
        )

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(self.n_joints, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # Residual position targets
        target = self.default_act_qpos + self.cfg.action_scale * action

        # Clip to actuator ctrl ranges
        for local_i, global_act_i in enumerate(self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[global_act_i]
            target[local_i] = float(np.clip(target[local_i], lo, hi))

        # Step the simulation at the higher frequency
        for _ in range(self.cfg.control_decimation):
            self.data.ctrl[self.act_ids] = target
            mujoco.mj_step(self.model, self.data)

        self._maybe_push()

        reward, info = self._compute_reward(action)
        terminated = self._is_terminated()
        self._step_count += 1
        truncated = self._step_count >= self.cfg.max_episode_steps

        self._prev_action = action.copy()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    # ─────────────────────────── Curriculum hook ─────────────────────────────

    def update_config(self, **kwargs):
        """Called by the curriculum callback via vec_env.env_method()."""
        for key, val in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, val)
            else:
                raise AttributeError(f"EnvConfig has no attribute '{key}'")
