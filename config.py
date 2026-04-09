from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    model_path: str = "mujoco_menagerie/unitree_g1/scene.xml"

    # ── Simulation timing ────────────────────────────────────────────────────
    # MuJoCo integrates at 5 ms; policy acts every 4 sim-steps → 50 Hz control.
    sim_timestep: float = 0.005
    control_decimation: int = 4    # policy dt = 0.005 × 4 = 0.02 s (50 Hz)
    max_episode_steps: int = 1000  # 20 s per episode

    # ── Standing target ──────────────────────────────────────────────────────
    # Pelvis height in G1's default standing keyframe is ~0.78 m.
    target_height: float = 0.78
    min_height: float = 0.40       # below this → episode terminates
    max_tilt_deg: float = 40.0     # torso tilt beyond this → episode terminates

    # ── Actions ──────────────────────────────────────────────────────────────
    # Policy outputs residuals in [-1, 1] scaled to ±action_scale radians.
    action_scale: float = 0.3

    # ── Reward weights ───────────────────────────────────────────────────────
    w_upright: float = 3.0         # main balance objective
    w_height: float = 2.0          # keep CoM near target height
    height_sigma: float = 5.0      # Gaussian sharpness for height reward
    w_contact: float = 1.0         # both feet on the ground
    w_joint_vel: float = 0.001     # penalise fast joint motion
    w_action_rate: float = 0.01    # penalise jerky actions
    w_energy: float = 0.0001       # penalise high torques
    w_survival: float = 0.5        # reward per step while upright

    # ── Disturbance (push) ───────────────────────────────────────────────────
    # Stage 1 starts with zero push; curriculum increases these.
    push_vel_max: float = 0.0      # max horizontal velocity impulse (m/s)
    push_interval_steps: int = 100 # apply push every N control steps (~2 s)

    # ── Domain randomisation ─────────────────────────────────────────────────
    use_domain_rand: bool = False
    mass_rand_ratio: float = 0.15   # ±15 % body mass
    friction_rand_ratio: float = 0.30
    damping_rand_ratio: float = 0.30

    # ── Episode initialisation ───────────────────────────────────────────────
    init_noise: float = 0.05       # uniform noise added to qpos at reset (rad)


@dataclass
class TrainConfig:
    # ── PPO hyperparameters ──────────────────────────────────────────────────
    total_timesteps: int = 50_000_000
    n_envs: int = 16               # parallel subprocess envs; good for RTX 3060
    n_steps: int = 2048            # rollout steps per env per update
    batch_size: int = 512
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Policy network ───────────────────────────────────────────────────────
    net_arch: list = field(default_factory=lambda: [512, 256, 128])

    # ── Logging & checkpoints ────────────────────────────────────────────────
    log_dir: str = "logs"
    save_dir: str = "models"
    save_freq: int = 500_000       # env-steps between checkpoints
    eval_freq: int = 500_000       # env-steps between eval runs
    n_eval_episodes: int = 5

    # ── Curriculum stage thresholds (total env-steps) ────────────────────────
    # Stage 1 → 2: enable small pushes
    # Stage 2 → 3: increase push magnitude
    # Stage 3 → 4: full domain randomisation
    curriculum_thresholds: list = field(default_factory=lambda: [
        2_000_000,
        10_000_000,
        25_000_000,
    ])
