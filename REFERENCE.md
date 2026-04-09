# MuJoCo Tutorial Reference

Quick-reference for the 5 notebooks in this folder.  
Notebooks: `tutorial.ipynb`, `mjspec.ipynb`, `LQR.ipynb`, `rollout.ipynb`, `least_squares.ipynb`

---

## Table of Contents
1. [Core Setup](#1-core-setup)
2. [Model & Data Basics](#2-model--data-basics)
3. [Simulation Loop & Rendering](#3-simulation-loop--rendering)
4. [Cameras](#4-cameras)
5. [Contacts & Forces](#5-contacts--forces)
6. [LQR Controller](#6-lqr-controller)
7. [mjSpec — Procedural Model Editing](#7-mjspec--procedural-model-editing)
8. [Rollout & Batch Simulation](#8-rollout--batch-simulation)
9. [Least Squares / Inverse Kinematics](#9-least-squares--inverse-kinematics)

---

## 1. Core Setup

```python
import mujoco
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import scipy.linalg

np.set_printoptions(precision=3, suppress=True, linewidth=100)
```

**Minimal model from XML string:**
```python
xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)
```

**Load from file:**
```python
model = mujoco.MjModel.from_xml_path('path/to/model.xml')
data  = mujoco.MjData(model)
```

---

## 2. Model & Data Basics

```python
# Counts
model.ngeom   # number of geoms
model.nbody   # number of bodies
model.njnt    # number of joints
model.nv      # degrees of freedom (velocity dim)
model.nq      # position dim (can be > nv due to quaternions)
model.nu      # number of actuators

# Named access — returns a NamedAccess object
model.geom('floor').rgba        # geom color
model.body('torso').id          # integer id
model.joint('knee').dofadr[0]   # first DoF address
model.actuator(0).trnid[0]      # joint id for actuator 0

# Data (simulation state)
data.qpos       # joint positions  (nq,)
data.qvel       # joint velocities (nv,)
data.ctrl       # actuator controls (nu,)
data.time       # simulation time

# Derived quantities (populated after mj_forward / mj_step)
data.xpos           # body positions in world frame (nbody, 3)
data.xquat          # body quaternions
data.geom_xpos      # geom positions
data.sensordata     # sensor readings
data.qfrc_actuator  # actuator forces in joint space
data.qfrc_inverse   # inverse dynamics result

# Reset helpers
mujoco.mj_resetData(model, data)
mujoco.mj_resetDataKeyframe(model, data, key_index)  # load saved keyframe
```

---

## 3. Simulation Loop & Rendering

```python
renderer = mujoco.Renderer(model)               # default 480x480
renderer = mujoco.Renderer(model, width=1280, height=720)

# Forward kinematics only (no stepping)
mujoco.mj_forward(model, data)

# Step simulation by one timestep (model.opt.timestep)
mujoco.mj_step(model, data)

# Render a frame
renderer.update_scene(data)
pixels = renderer.render()          # numpy RGB array (H, W, 3)

# Show single image
media.show_image(pixels)

# Render a video
DURATION  = 3    # seconds
FRAMERATE = 60   # Hz
frames = []
while data.time < DURATION:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * FRAMERATE:
        renderer.update_scene(data)
        frames.append(renderer.render())
media.show_video(frames, fps=FRAMERATE)
```

**Render modes:**
```python
renderer.enable_depth_rendering()          # depth image
renderer.enable_segmentation_rendering()   # segmentation labels
renderer.disable_depth_rendering()         # back to RGB
```

**Scene options (visualisation flags):**
```python
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
model.vis.map.force = 0.01   # scale: 1cm per Newton
renderer.update_scene(data, camera, scene_option)
```

---

## 4. Cameras

```python
# Free camera
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance  = 2.3
camera.azimuth   = 100       # degrees
camera.elevation = -20
camera.lookat    = data.body('torso').subtree_com  # track CoM

renderer.update_scene(data, camera)

# Fixed (model-defined) camera by name
renderer.update_scene(data, "fixed_cam_name")

# Camera intrinsic matrix from model
width, height = renderer.width, renderer.height
fovy_rad = np.deg2rad(model.vis.global_.fovy)
f = (height / 2) / np.tan(fovy_rad / 2)
K = np.array([[f, 0, width/2],
              [0, f, height/2],
              [0, 0, 1]])
```

---

## 5. Contacts & Forces

```python
# Number of active contacts
data.ncon

# Iterate contacts
for i, contact in enumerate(data.contact):
    print(contact.pos)      # contact position (3,)
    print(contact.frame)    # contact frame (9,) — row-major 3x3
    print(contact.dist)     # penetration distance

# Jacobian for a body
jacp = np.zeros((3, model.nv))   # position Jacobian
jacr = np.zeros((3, model.nv))   # rotation Jacobian
mujoco.mj_jac(model, data, jacp, jacr, point, body_id)

# Subtree CoM Jacobian
jac_com = np.zeros((3, model.nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)

# Body CoM Jacobian
jac_foot = np.zeros((3, model.nv))
mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body('foot_left').id)

# Site Jacobian (used in IK)
jac_site = np.zeros((6, model.nv))   # 3 pos + 3 rot rows
mujoco.mj_jacSite(model, data, jac_site[:3], jac_site[3:], site_id)
```

---

## 6. LQR Controller

Full pipeline from `LQR.ipynb`:

### 6a. Find control setpoint via inverse dynamics
```python
# Load keyframe, run forward, assert zero acceleration
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0

# Scan height offsets to find minimal "magic" vertical force
height_offsets = np.linspace(-0.001, 0.001, 2001)
vertical_forces = []
for offset in height_offsets:
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    data.qpos[2] += offset
    mujoco.mj_inverse(model, data)
    vertical_forces.append(data.qfrc_inverse[2])

best_offset = height_offsets[np.argmin(np.abs(vertical_forces))]

# Apply best offset, save setpoint
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()

# ctrl setpoint via actuator moment arm pseudo-inverse
actuator_moment = np.zeros((model.nu, model.nv))
mujoco.mju_sparse2dense(
    actuator_moment,
    data.actuator_moment.reshape(-1),
    data.moment_rownnz,
    data.moment_rowadr,
    data.moment_colind.reshape(-1),
)
ctrl0 = (np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)).flatten()
```

### 6b. Build Q matrix
```python
nv, nu = model.nv, model.nu

# Balancing cost: keep CoM over foot
jac_com  = np.zeros((3, nv))
jac_foot = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com,  model.body('torso').id)
mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body('foot_left').id)
Qbalance = (jac_com - jac_foot).T @ (jac_com - jac_foot)

# Joint cost with per-group weights
joint_names   = [model.joint(i).name for i in range(model.njnt)]
root_dofs     = range(6)
body_dofs     = range(6, nv)
abdomen_dofs  = [model.joint(n).dofadr[0] for n in joint_names if 'abdomen' in n and 'z' not in n]
left_leg_dofs = [model.joint(n).dofadr[0] for n in joint_names
                 if 'left' in n and any(p in n for p in ('hip','knee','ankle')) and 'z' not in n]
balance_dofs  = abdomen_dofs + left_leg_dofs
other_dofs    = np.setdiff1d(body_dofs, balance_dofs)

Qjoint = np.eye(nv)
Qjoint[root_dofs,    root_dofs]    *= 0
Qjoint[balance_dofs, balance_dofs] *= 3
Qjoint[other_dofs,   other_dofs]   *= 0.3
Qpos = 1000 * Qbalance + Qjoint

Q = np.block([[Qpos,           np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])
R = np.eye(nu)
```

### 6c. Linearise & solve Riccati
```python
# Finite-difference transition matrices A, B
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0
A = np.zeros((2*nv, 2*nv))
B = np.zeros((2*nv, nu))
mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)

# Solve discrete-time Riccati equation
P = scipy.linalg.solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
```

### 6d. Apply LQR in simulation loop
```python
dq = np.zeros(model.nv)
while data.time < DURATION:
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel))
    data.ctrl = ctrl0 - K @ dx
    mujoco.mj_step(model, data)
```

---

## 7. mjSpec — Procedural Model Editing

```python
import mujoco

# Parse existing XML into a spec
spec = mujoco.MjSpec.from_file('model.xml')
# or
spec = mujoco.MjSpec()

# Compile spec → model
model = spec.compile()
data  = mujoco.MjData(model)

# Add a body
body = spec.worldbody.add_body()
body.name = "my_body"
body.pos  = [0, 0, 1]

# Add a geom to a body
geom = body.add_geom()
geom.type    = mujoco.mjtGeom.mjGEOM_BOX
geom.size    = [0.1, 0.1, 0.1]
geom.rgba    = [1, 0, 0, 1]

# Add a joint
joint = body.add_joint()
joint.type = mujoco.mjtJoint.mjJNT_FREE

# Add a mesh asset
mesh = spec.add_mesh()
mesh.name    = "my_mesh"
mesh.file    = "mesh.stl"

# Add a heightfield
hfield = spec.add_hfield()
hfield.name    = "terrain"
hfield.nrow    = 64
hfield.ncol    = 64
hfield.size    = [10, 10, 1, 0.1]
# data is a flat float array of nrow*ncol values in [0,1]
hfield.data    = terrain_data.flatten()

# Modify an existing element by name
spec.find_body('torso').pos[2] += 0.1

# Recompile after changes
model = spec.compile()
```

**Procedural terrain pattern:**
```python
def make_terrain(nrow, ncol):
    data = np.zeros((nrow, ncol))
    # ... fill with Perlin noise or geometric patterns ...
    return data.flatten().astype(np.float32)
```

---

## 8. Rollout & Batch Simulation

```python
import mujoco
from mujoco import rollout   # multi-threaded CPU rollout

# Single rollout
initial_state = np.concatenate([data.qpos, data.qvel])
nstep = 200
states, sensordata = rollout.rollout(model, data, initial_state, nstep)

# Batch rollout — vary initial conditions across N episodes
N = 100
initial_states = np.tile(initial_state, (N, 1))
initial_states[:, 0] += np.random.uniform(-0.1, 0.1, N)   # vary one DoF

# rollout.rollout returns (nstep, N, state_dim) and (nstep, N, sensor_dim)
states, sensordata = rollout.rollout(model, data, initial_states, nstep)

# State helper — pack/unpack qpos+qvel
def get_state(data):
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])

def set_state(model, data, state):
    nq = model.nq
    data.qpos[:] = state[:nq]
    data.qvel[:] = state[nq:]
    mujoco.mj_forward(model, data)

# Branching: clone data at a checkpoint
import copy
data_checkpoint = copy.deepcopy(data)   # or use rollout's clone API
```

**MJX (JAX-based GPU rollout):**
```python
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp

mx = mjx.put_model(model)
dx = mjx.put_data(model, data)

# JIT-compiled step
jit_step = jax.jit(mjx.step)

# Batched step with vmap
batched_step = jax.vmap(mjx.step, in_axes=(None, 0))
dx_batch = jax.vmap(mjx.put_data, in_axes=(None, 0))(model, [data]*N)
dx_batch = batched_step(mx, dx_batch)
```

---

## 9. Least Squares / Inverse Kinematics

```python
from mujoco import minimize   # MuJoCo's built-in nonlinear least squares

# --- Rosenbrock example (unconstrained) ---
def rosenbrock_residual(x):
    return np.array([10*(x[1] - x[0]**2), (1 - x[0])])

result = minimize.least_squares(rosenbrock_residual, x0=np.array([-1.0, 1.0]))
print(result.x)

# --- Box-constrained ---
result = minimize.least_squares(
    residual_fn,
    x0=x0,
    bounds=(lower, upper)   # arrays of same shape as x0
)

# --- Panda / Humanoid IK ---
def ik_residual(ctrl, model, data, site_id, target_pos, target_quat):
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    site_pos  = data.site_xpos[site_id].copy()
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
    pos_err  = site_pos - target_pos
    quat_err = np.zeros(3)
    mujoco.mju_subQuat(quat_err, target_quat, site_quat)
    return np.concatenate([pos_err, quat_err])

def ik_jacobian(ctrl, model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])

# Warmstarting across frames
prev_ctrl = ctrl0.copy()
for target in targets:
    result = minimize.least_squares(
        lambda c: ik_residual(c, model, data, site_id, target, target_quat),
        x0=prev_ctrl,
        bounds=(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    )
    prev_ctrl = result.x
```

**Key IK helpers:**
```python
# Convert 3x3 rotation matrix → quaternion
quat = np.zeros(4)
mujoco.mju_mat2Quat(quat, mat_3x3_row_major)

# Quaternion difference (tangent space, 3-vector)
err = np.zeros(3)
mujoco.mju_subQuat(err, target_quat, current_quat)

# Position difference accounting for quaternion DoFs
dq = np.zeros(model.nv)
mujoco.mj_differentiatePos(model, dq, dt, qpos_ref, qpos_current)
```

---

## Quick API Cheat Sheet

| Task | Function |
|------|----------|
| Forward kinematics | `mujoco.mj_forward(model, data)` |
| Step simulation | `mujoco.mj_step(model, data)` |
| Inverse dynamics | `mujoco.mj_inverse(model, data)` |
| Transition Jacobians (A,B) | `mujoco.mjd_transitionFD(model, data, eps, centered, A, B, C, D)` |
| Position difference | `mujoco.mj_differentiatePos(model, dq, dt, qpos1, qpos2)` |
| Body Jacobian | `mujoco.mj_jac(model, data, jacp, jacr, point, body_id)` |
| SubtreeCoM Jacobian | `mujoco.mj_jacSubtreeCom(model, data, jac, body_id)` |
| Site Jacobian | `mujoco.mj_jacSite(model, data, jacp, jacr, site_id)` |
| Sparse → dense matrix | `mujoco.mju_sparse2dense(out, vals, rownnz, rowadr, colind)` |
| Quaternion multiply | `mujoco.mju_mulQuat(res, q1, q2)` |
| Quat → rotation mat | `mujoco.mju_quat2Mat(mat, quat)` |
| Rotation mat → quat | `mujoco.mju_mat2Quat(quat, mat)` |
| Quat difference | `mujoco.mju_subQuat(err, q1, q2)` |
| Discrete Riccati | `scipy.linalg.solve_discrete_are(A, B, Q, R)` |
| LQR gain | `K = inv(R + B'PB) @ B'PA` |
