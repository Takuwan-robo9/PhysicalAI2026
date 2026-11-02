"""MuJoCo 上の Franka Panda を動かすための環境・制御モジュール.

- 逆運動学 (減衰最小二乗法, Damped Least Squares) で手先の目標位置・姿勢を関節角に変換
- 最小躍度 (minimum-jerk) 補間で滑らかな軌道を生成
- ピック&プレースの動作プリミティブを提供
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "assets" / "franka_emika_panda" / "tidy_scene.xml"

HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, -0.7853])
TCP_OFFSET = 0.1034          # hand 原点から指先中心までの距離 [m]
GRIP_OPEN, GRIP_CLOSE, GRIP_RELEASE = 255.0, 0.0, 170.0
MOVABLE = ["red_cube", "yellow_cube", "blue_cube", "green_cylinder"]


def rot_down(yaw: float) -> np.ndarray:
    """手先が真下を向き、指の開閉方向 (hand y 軸) が yaw だけ回った姿勢の回転行列."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class TidyEnv:
    def __init__(self, frame_callback: Callable[["TidyEnv"], None] | None = None, fps: int = 30):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)  # IK 計算専用 (シミュレーション状態を汚さない)
        self.hand_id = self.model.body("hand").id
        self.arm_qadr = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dadr = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.q_lo = self.model.jnt_range[[self.model.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.q_hi = self.model.jnt_range[[self.model.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self.frame_callback = frame_callback
        self.fps = fps
        self.q_cmd = HOME_Q.copy()
        self.grip_cmd = GRIP_OPEN
        self.status = ""   # 動画に表示する現在の動作
        self.reset()

    # ------------------------------------------------------------------ 基本
    def reset(self, layout: dict[str, tuple[float, float, float]] | None = None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_Q
        self.data.qpos[7:9] = 0.04
        if layout:
            for name, (x, y, yaw) in layout.items():
                adr = self.model.joint(name).qposadr[0]
                z = self.model.geom(name).size[2] if self.model.geom(name).type == mujoco.mjtGeom.mjGEOM_BOX \
                    else self.model.geom(name).size[1]
                self.data.qpos[adr:adr + 3] = [x, y, z]
                self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.q_cmd = HOME_Q.copy()
        self.grip_cmd = GRIP_OPEN
        self._apply_ctrl()
        mujoco.mj_forward(self.model, self.data)
        self._next_frame_t = 0.0

    def _apply_ctrl(self):
        self.data.ctrl[:7] = self.q_cmd
        self.data.ctrl[7] = self.grip_cmd

    def step(self, seconds: float):
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(n):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
            if self.frame_callback and self.data.time >= self._next_frame_t:
                self._next_frame_t += 1.0 / self.fps
                self.frame_callback(self)

    def tcp_pose(self, data: mujoco.MjData | None = None):
        d = data or self.data
        R = d.xmat[self.hand_id].reshape(3, 3)
        return d.xpos[self.hand_id] + R[:, 2] * TCP_OFFSET, R

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    # ------------------------------------------------------------------ IK
    def solve_ik(self, pos, R_des, q_init, iters: int = 60, tol: float = 1e-4) -> np.ndarray:
        """減衰最小二乗法による 6 自由度 IK (位置 + 姿勢)."""
        q = q_init.copy()
        kin = self.kin
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            kin.qpos[:] = self.data.qpos
            kin.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, kin)
            mujoco.mj_comPos(self.model, kin)
            p, R = self.tcp_pose(kin)
            e_pos = pos - p
            e_rot = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            err = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jac(self.model, kin, jacp, jacr, p, self.hand_id)
            J = np.vstack([jacp[:, self.arm_dadr], jacr[:, self.arm_dadr]])
            lam = 1e-2
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
            # 冗長自由度: ホーム姿勢へ引き寄せる (零空間)
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (HOME_Q - q))
            q = np.clip(q + dq, self.q_lo + 1e-3, self.q_hi - 1e-3)
        return q

    # ------------------------------------------------------------------ 動作プリミティブ
    def move_to(self, pos, yaw: float, duration: float, ctrl_dt: float = 0.01):
        """現在の指令姿勢から目標まで、デカルト空間で min-jerk 補間しながら移動."""
        p0, R0 = self.tcp_pose_cmd()
        yaw0 = self._yaw_of(R0)
        dyaw = (yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi
        pos = np.asarray(pos, dtype=float)
        n = max(1, int(duration / ctrl_dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            p = p0 + s * (pos - p0)
            self.q_cmd = self.solve_ik(p, rot_down(yaw0 + s * dyaw), self.q_cmd, iters=8)
            self.step(ctrl_dt)
        # 最終姿勢を精密に
        self.q_cmd = self.solve_ik(pos, rot_down(yaw), self.q_cmd)

    def move_joints(self, q_goal, duration: float, ctrl_dt: float = 0.01):
        q0 = self.q_cmd.copy()
        n = max(1, int(duration / ctrl_dt))
        for k in range(1, n + 1):
            self.q_cmd = q0 + min_jerk(k / n) * (q_goal - q0)
            self.step(ctrl_dt)

    def tcp_pose_cmd(self):
        self.kin.qpos[:] = self.data.qpos
        self.kin.qpos[self.arm_qadr] = self.q_cmd
        mujoco.mj_kinematics(self.model, self.kin)
        return self.tcp_pose(self.kin)

    @staticmethod
    def _yaw_of(R) -> float:
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def gripper(self, close: bool, wait: float = 0.6):
        self.grip_cmd = GRIP_CLOSE if close else GRIP_OPEN
        self.step(wait)

    def finger_gap(self) -> float:
        return float(self.data.qpos[7] + self.data.qpos[8])

    def go_home(self, duration: float = 1.5):
        self.status = "ホーム姿勢へ戻る"
        self.move_joints(HOME_Q, duration)

    def pick_and_place(self, pick_xyz, pick_yaw, place_xyz, place_yaw=None,
                       hover: float = 0.14, speed: float = 1.0) -> bool:
        """pick_xyz: 把持点 (指先中心を合わせる位置), place_xyz: 置いたときの把持点の位置.
        持ち上げた時点で物体を保持できているか (指が閉じ切っていないか) を返す."""
        place_yaw = pick_yaw if place_yaw is None else place_yaw
        pick_xyz, place_xyz = np.asarray(pick_xyz, float), np.asarray(place_xyz, float)
        t = 1.0 / speed
        lift = max(hover, place_xyz[2] - pick_xyz[2] + 0.1)   # 積む場合は高めに持ち上げる
        self.status = "物体の上へ移動"
        self.move_to(pick_xyz + [0, 0, hover], pick_yaw, 1.6 * t)
        self.status = "降下"
        self.move_to(pick_xyz, pick_yaw, 1.0 * t)
        self.status = "把持"
        self.gripper(True, 0.7)
        self.status = "持ち上げ"
        self.move_to(pick_xyz + [0, 0, lift], pick_yaw, 1.0 * t)
        held = self.finger_gap() > 0.005
        self.status = "置き場所へ移動"
        above = np.array([place_xyz[0], place_xyz[1], pick_xyz[2] + lift])
        self.move_to(above, place_yaw, 1.8 * t)
        self.status = "そっと置く"
        self.move_to(place_xyz + [0, 0, 0.004], place_yaw, 1.2 * t)
        # 隣の物体に指が当たらないよう、少しだけ開いて離す (指の間隔 約 5.3cm)
        self.grip_cmd = GRIP_RELEASE
        self.step(0.5)
        self.status = "離脱"
        self.move_to(place_xyz + [0, 0, hover * 0.8], place_yaw, 0.9 * t)
        self.grip_cmd = GRIP_OPEN
        return held
