"""真上カメラの RGB-D 画像から物体を認識するモジュール.

シミュレータの「正解位置」は一切使わず、レンダリングした RGB 画像と深度画像だけから
  1. 深度 → 3 次元点群 (テーブル座標系の高さマップ) に逆投影
  2. HSV 色空間で色ごとに領域分割
  3. 高さで「ブロック (背が高い)」と「お皿 (平たい)」を分離
  4. 上面の形状 (外接矩形への充填率) で立方体 / 円柱を判別、向き (yaw) を推定
を行う.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

# OpenCV の H は 0-180
COLOR_RANGES = {
    "red": [(0, 9), (172, 180)],
    "yellow": [(18, 35)],
    "green": [(40, 85)],
    "blue": [(95, 130)],
    "pink": [(140, 171)],
}
COLOR_JA = {"red": "赤", "yellow": "黄色", "green": "緑", "blue": "青", "pink": "ピンク"}
SHAPE_JA = {"cube": "立方体のブロック", "cylinder": "円柱のブロック", "plate": "お皿"}
DRAW_BGR = {"red": (40, 40, 230), "yellow": (0, 210, 240), "green": (60, 190, 60),
            "blue": (230, 120, 30), "pink": (180, 80, 240)}


@dataclass
class Detection:
    id: str
    color: str
    kind: str                  # "block" | "plate"
    shape: str                 # "cube" | "cylinder" | "plate"
    xy: np.ndarray             # 中心 (world, m)
    top_z: float               # 上面の高さ
    support_z: float = 0.0     # 置かれている面の高さ
    yaw: float = 0.0           # 把持向き (立方体の辺の向き)
    radius: float = 0.0        # お皿の半径
    contour_px: np.ndarray | None = field(default=None, repr=False)

    @property
    def height(self) -> float:
        return self.top_z - self.support_z

    def label_ja(self) -> str:
        return f"{COLOR_JA[self.color]}の{SHAPE_JA[self.shape]}"


class Perception:
    def __init__(self, model: mujoco.MjModel, cam: str = "top_cam", width: int = 640, height: int = 480):
        self.model = model
        self.cam_id = model.camera(cam).id
        self.W, self.H = width, height
        self.rgb_r = mujoco.Renderer(model, height, width)
        self.depth_r = mujoco.Renderer(model, height, width)
        self.depth_r.enable_depth_rendering()
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        self.f = 0.5 * height / np.tan(fovy / 2)
        self.last_rgb = None
        self.last_dets: list[Detection] = []

    # -------------------------------------------------------------- 撮影 + 逆投影
    def capture(self, data: mujoco.MjData):
        self.rgb_r.update_scene(data, camera=self.cam_id)
        rgb = self.rgb_r.render().copy()
        self.depth_r.update_scene(data, camera=self.cam_id)
        depth = self.depth_r.render().copy()
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        pc = np.stack([(u - self.W / 2) * depth / self.f,
                       -(v - self.H / 2) * depth / self.f,
                       -depth], axis=-1)                         # カメラ座標系
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        pw = pc @ R.T + data.cam_xpos[self.cam_id]               # world 座標系 (H, W, 3)
        return rgb, pw

    # -------------------------------------------------------------- 検出
    def detect(self, data: mujoco.MjData) -> list[Detection]:
        rgb, pw = self.capture(data)
        z = pw[..., 2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat_ok = (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
        on_table = (pw[..., 0] > 0.0) & (z > -0.01)
        dets: list[Detection] = []
        for color, ranges in COLOR_RANGES.items():
            hmask = np.zeros(z.shape, bool)
            for lo, hi in ranges:
                hmask |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
            cmask = hmask & sat_ok & on_table
            high = (cmask & (z > 0.016)).astype(np.uint8)
            low = (cmask & (z > 0.003) & (z <= 0.016)).astype(np.uint8)
            dets += self._blocks(color, high, pw)
            dets += self._plates(color, low, pw)
        # 置かれている面の高さ (お皿の上 / 他ブロックの上 / テーブル)
        for d in dets:
            if d.kind == "block":
                d.support_z = self._support_height(d, pw)
        self.last_rgb, self.last_dets = rgb, dets
        return dets

    def _blocks(self, color, mask, pw):
        out = []
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 40:
                continue
            comp = lab == i
            zs = pw[..., 2][comp]
            top = float(np.percentile(zs, 90))
            top_mask = comp & (pw[..., 2] > top - 0.006)        # 上面だけ取り出す
            if top_mask.sum() < 150:      # 上面が見えていない (積まれた物の下の側面だけ) → 無視
                continue
            pts = pw[top_mask][:, :2]
            xy = pts.mean(axis=0)
            # 形状判定: 上面の点群の最小外接矩形への充填率 (正方形≈1.0, 円≈0.785)
            pts_mm = (pts * 1000).astype(np.float32)
            rect = cv2.minAreaRect(pts_mm)
            (w, h) = rect[1]
            fill = cv2.contourArea(cv2.convexHull(pts_mm)) / max(w * h, 1e-6)
            shape = "cube" if fill > 0.88 else "cylinder"
            yaw = np.deg2rad(rect[2]) if shape == "cube" else 0.0
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out.append(Detection(id=f"{color}_{shape}", color=color, kind="block", shape=shape, xy=xy,
                                 top_z=top, yaw=float(yaw), contour_px=max(cnts, key=cv2.contourArea)))
        return out

    def _plates(self, color, mask, pw):
        out = []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 1500:      # 小さい領域はブロックの側面など
                continue
            comp = lab == i
            pts = pw[comp]
            (cx, cy), r = cv2.minEnclosingCircle((pts[:, :2] * 1000).astype(np.float32))
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out.append(Detection(id=f"{color}_plate", color=color, kind="plate", shape="plate",
                                 xy=np.array([cx, cy]) / 1000, top_z=float(np.median(pts[:, 2])),
                                 radius=r / 1000, contour_px=max(cnts, key=cv2.contourArea)))
        return out

    def _support_height(self, d: Detection, pw) -> float:
        """物体の周囲 (半径 3.5〜5cm のリング) の高さから、置かれている面の高さを推定."""
        dist = np.linalg.norm(pw[..., :2] - d.xy, axis=-1)
        ring = (dist > 0.035) & (dist < 0.05)
        zs = pw[..., 2][ring]
        zs = zs[zs < d.top_z - 0.01]
        return float(np.percentile(zs, 30)) if zs.size else 0.0

    # -------------------------------------------------------------- 可視化
    def annotate(self, dets: list[Detection] | None = None, rgb=None) -> np.ndarray:
        """検出結果を描いた BGR 画像を返す."""
        dets = self.last_dets if dets is None else dets
        img = cv2.cvtColor(self.last_rgb if rgb is None else rgb, cv2.COLOR_RGB2BGR)
        for d in dets:
            if d.contour_px is None:
                continue
            col = DRAW_BGR[d.color]
            cv2.drawContours(img, [d.contour_px], -1, (255, 255, 255), 4)
            cv2.drawContours(img, [d.contour_px], -1, col, 2)
            x, y, w, h = cv2.boundingRect(d.contour_px)
            cv2.putText(img, d.id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, d.id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img
