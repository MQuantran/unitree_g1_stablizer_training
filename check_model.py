"""
check_model.py — Inspect the G1 MJCF model before training.

Run this first to verify that all joint and actuator names used in
G1StandEnv.ACT_JOINT_NAMES actually exist in the loaded model.

Usage:
  python check_model.py
  python check_model.py --model path/to/scene.xml
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from config import EnvConfig


def check(model_path: str):
    print(f"Loading: {model_path}\n")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  nq (pos dofs)   : {model.nq}")
    print(f"  nv (vel dofs)   : {model.nv}")
    print(f"  nu (actuators)  : {model.nu}")
    print(f"  nbody           : {model.nbody}")
    print(f"  njnt            : {model.njnt}")
    print(f"  ngeom           : {model.ngeom}")
    print(f"  nkey (keyframes): {model.nkey}")
    print("=" * 60)

    # ── Joints ────────────────────────────────────────────────────────────────
    print("\nJOINTS")
    print(f"  {'id':>4}  {'name':<40}  {'type':>5}  {'qposadr':>8}  {'dofadr':>7}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*5}  {'-'*8}  {'-'*7}")
    jtype_map = {
        mujoco.mjtJoint.mjJNT_FREE: "free",
        mujoco.mjtJoint.mjJNT_BALL: "ball",
        mujoco.mjtJoint.mjJNT_SLIDE: "slide",
        mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    }
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        jtype = jtype_map.get(model.jnt_type[i], str(model.jnt_type[i]))
        print(
            f"  {i:>4}  {name:<40}  {jtype:>5}"
            f"  {model.jnt_qposadr[i]:>8}  {model.jnt_dofadr[i]:>7}"
        )

    # ── Actuators ─────────────────────────────────────────────────────────────
    print("\nACTUATORS")
    print(f"  {'id':>4}  {'name':<40}  {'ctrlrange'}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*20}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  {i:>4}  {name:<40}  [{lo:+.3f}, {hi:+.3f}]")

    # ── Bodies ────────────────────────────────────────────────────────────────
    print("\nBODIES (id | name)")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        print(f"  {i:>4}  {name}")

    # ── Keyframes ─────────────────────────────────────────────────────────────
    if model.nkey > 0:
        print("\nKEYFRAMES")
        for k in range(model.nkey):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}"
            mujoco.mj_resetDataKeyframe(model, data, k)
            mujoco.mj_forward(model, data)
            # Root body CoM height as a sanity check
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            h = data.subtree_com[pelvis_id, 2] if pelvis_id != -1 else float("nan")
            print(f"  {k}: '{name}'  — pelvis height = {h:.4f} m")
    else:
        print("\nNo keyframes found — default pose will be used.")

    # ── Validate G1StandEnv joint/actuator names ──────────────────────────────
    from envs.g1_stand_env import G1StandEnv

    print(f"\nVALIDATING G1StandEnv.ACT_JOINT_NAMES ({len(G1StandEnv.ACT_JOINT_NAMES)} joints)")
    print(f"  {'joint name':<40}  {'jnt_id':>7}  {'act_name':<40}  {'act_id':>7}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*40}  {'-'*7}")

    all_ok = True
    for jnt_name in G1StandEnv.ACT_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)

        # Try both actuator name conventions
        act_name_try = jnt_name.replace("_joint", "")
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name_try)
        if act_id == -1:
            act_name_try = jnt_name
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name_try)

        ok = jnt_id != -1 and act_id != -1
        flag = "OK " if ok else "MISSING"
        if not ok:
            all_ok = False
        print(
            f"  {jnt_name:<40}  {jnt_id:>7}  {act_name_try:<40}  {act_id:>7}  {flag}"
        )

    print()
    if all_ok:
        print("All joints and actuators resolved successfully.")
    else:
        print(
            "Some names could not be resolved.\n"
            "Update G1StandEnv.ACT_JOINT_NAMES / TORSO_BODY / LEFT_FOOT_BODY / "
            "RIGHT_FOOT_BODY to match the names printed above."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=EnvConfig().model_path,
        metavar="PATH",
        help="Path to MJCF scene.xml (default: config.EnvConfig.model_path)",
    )
    args = parser.parse_args()
    check(args.model)
