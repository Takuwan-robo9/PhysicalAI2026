"""言葉で指示できる片付けロボット: 認識 → LLM 計画 → 検証/修復 → 実行 → 再認識で確認.

TidyRobotApp.run_command(instruction) が 1 つの指示を処理する一連のループ.
動画フレームは VideoComposer が合成する (メインカメラ + 字幕パネル + 認識結果の小窓).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception import Detection, Perception
from planner import LLMPlanner, object_name_ja, plan_to_ja
from robot import HOME_Q, MOVABLE, TidyEnv, rot_down

OBS_Q = np.array([0.0, -1.3, 0.0, -2.9, 0.0, 1.9, 0.785])   # カメラの視野からアームを退避させる観測姿勢
TABLE_AREA = ((0.45, 0.68), (-0.16, 0.16))                     # 退避先として使うテーブル上の領域

FONT_CANDIDATES = [
    "C:/Windows/Fonts/BIZ-UDGothicB.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
]


def wrap_quarter(a: float) -> float:
    """角度を [-45°, 45°) に折り返す (立方体は 90° 対称)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


# ============================================================================ 世界モデル
class WorldModel:
    """認識結果 + 記憶 から「どの物体が何の上にあるか」を管理する.

    上から見えなくなった (上に物が積まれた) ブロックは記憶から補完する.
    """

    def __init__(self):
        self.mem: dict[str, Detection] = {}
        self.objects: dict[str, dict] = {}
        self.dets: dict[str, Detection] = {}
        self.heights: dict[str, float] = {}   # 初めて見たときの物体の高さ (形は変わらない)

    def update(self, dets: list[Detection]):
        visible = {d.id: d for d in dets}
        for d in dets:
            if d.kind != "block":
                continue
            if d.id not in self.heights:
                self.heights[d.id] = d.height
            else:   # 上面の高さ - 既知の物体高さ = 置かれている面の高さ (積み上げにも対応)
                d.support_z = d.top_z - self.heights[d.id]
        plates = {k: d for k, d in visible.items() if d.kind == "plate"}
        objects: dict[str, dict] = {}
        hidden_by: dict[str, str] = {}
        for k, d in visible.items():
            if d.kind == "plate":
                continue
            on = "table"
            if d.support_z > 0.004:
                plate = next((p for p in plates.values()
                              if np.linalg.norm(d.xy - p.xy) < p.radius and d.support_z < p.top_z + 0.01), None)
                if plate is not None:
                    on = plate.id
                else:  # 何かの上に積まれている → 真下にあったはずの (今は見えない) ブロック
                    below = next((m for m in self.mem.values() if m.kind == "block" and m.id not in visible
                                  and np.linalg.norm(m.xy - d.xy) < 0.03), None)
                    if below is not None:
                        on = below.id
                        hidden_by[below.id] = k
            objects[k] = {"color": d.color, "shape": d.shape, "on": on}
        for k, top in hidden_by.items():  # 見えないが下にあるブロック
            m = self.mem[k]
            prev_on = self.objects.get(k, {}).get("on", "table")
            objects[k] = {"color": m.color, "shape": m.shape, "on": prev_on, "hidden": top}
            visible[k] = m
        for k, d in plates.items():
            objects[k] = {"color": d.color, "shape": "plate"}
        # 表示順を安定させる
        self.objects = dict(sorted(objects.items(), key=lambda kv: (kv[1]["shape"] == "plate", kv[0])))
        self.dets = visible
        self.mem = dict(visible)


# ============================================================================ 動画合成
class VideoComposer:
    W, H = 1280, 720

    def __init__(self, env: TidyEnv, out_path: Path, fps: int = 30):
        self.env = env
        self.fps = fps
        self.out_path = Path(out_path)
        self.tmp_path = self.out_path.with_suffix(".tmp.mp4")
        self.writer = cv2.VideoWriter(str(self.tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.W, self.H))
        self.renderer = mujoco.Renderer(env.model, self.H, self.W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.45, 0.0, 0.06]
        self.cam.distance = 1.35
        self.cam.elevation = -34
        font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
        load = (lambda s: ImageFont.truetype(font_path, s)) if font_path else (lambda s: ImageFont.load_default())
        self.f_title, self.f_mid, self.f_small, self.f_tiny = load(46), load(22), load(19), load(15)
        self.inset = None
        # 字幕パネルの状態
        self.instruction = ""
        self.reply = ""
        self.interp = ""
        self.plan_lines: list[str] = []
        self.notes: list[str] = []
        self.current = -1
        self.done: set[int] = set()
        self.banner = ""
        self.model_label = ""
        self.n_frames = 0

    # -------------------------------------------------------------- フレーム生成
    def capture(self, env: TidyEnv | None = None):
        env = env or self.env
        self.cam.azimuth = 150 + 15 * np.sin(env.data.time / 9.0)
        self.renderer.update_scene(env.data, camera=self.cam)
        frame = self.renderer.render()
        self._write(self._overlay(frame))

    def hold(self, seconds: float):
        """シミュレーションを止めたまま (LLM 推論中など) フレームを出す."""
        for _ in range(int(seconds * self.fps)):
            self.capture()

    def card(self, title: str, lines: list[str], seconds: float):
        """タイトル / エンディング用の静止カード."""
        img = Image.new("RGB", (self.W, self.H), (18, 24, 34))
        d = ImageDraw.Draw(img)
        d.text((80, 150), title, font=self.f_title, fill=(255, 255, 255))
        y = 250
        for ln in lines:
            d.text((84, y), ln, font=self.f_mid, fill=(200, 215, 235))
            y += 40
        arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for _ in range(int(seconds * self.fps)):
            self._write(arr)

    def _write(self, bgr):
        self.writer.write(bgr)
        self.n_frames += 1

    def _overlay(self, rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(rgb).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # 左パネル
        texts = [(f"「{self.instruction}」", self.f_mid), (self.reply, self.f_small),
                 (f"解釈: {self.interp}", self.f_tiny)] + \
            [("▶ " + ln, self.f_small) for ln in self.plan_lines] + [(n, self.f_tiny) for n in self.notes]
        px, py = 18, 18
        pw = max([420] + [d.textlength(t, font=f) + 40 for t, f in texts])
        lines_h = 176 + 30 * (max(1, len(self.plan_lines)) + len(self.notes))
        d.rounded_rectangle([px, py, px + pw, py + lines_h], 14, fill=(10, 14, 22, 190))
        y = py + 12
        d.text((px + 16, y), "人間の指示", font=self.f_tiny, fill=(150, 170, 200, 255))
        y += 20
        d.text((px + 16, y), f"「{self.instruction}」" if self.instruction else "—", font=self.f_mid,
               fill=(255, 255, 255, 255))
        y += 36
        d.text((px + 16, y), f"AI ({self.model_label}) の返事", font=self.f_tiny, fill=(150, 170, 200, 255))
        y += 20
        d.text((px + 16, y), self.reply or "…", font=self.f_small, fill=(140, 230, 255, 255))
        y += 28
        if self.interp:
            d.text((px + 16, y), f"解釈: {self.interp}", font=self.f_tiny, fill=(170, 200, 220, 255))
        y += 32
        d.text((px + 16, y), "作業計画", font=self.f_tiny, fill=(150, 170, 200, 255))
        y += 22
        for i, ln in enumerate(self.plan_lines):
            col = (255, 220, 80, 255) if i == self.current else \
                (130, 210, 130, 255) if i in self.done else (225, 225, 225, 255)
            mark = "▶ " if i == self.current else ("✓ " if i in self.done else "   ")
            d.text((px + 16, y), mark + ln, font=self.f_small, fill=col)
            y += 30
        if not self.plan_lines:
            d.text((px + 16, y), "(なし)", font=self.f_small, fill=(180, 180, 180, 255))
            y += 30
        for n in self.notes:
            d.text((px + 16, y), n, font=self.f_tiny, fill=(255, 170, 90, 255))
            y += 30
        # 右上: 実行中の動作 / バナー
        if self.env.status:
            txt = f"動作: {self.env.status}"
            w = d.textlength(txt, font=self.f_small)
            d.rounded_rectangle([self.W - w - 50, 18, self.W - 18, 58], 10, fill=(10, 14, 22, 190))
            d.text((self.W - w - 34, 26), txt, font=self.f_small, fill=(255, 255, 255, 255))
        if self.banner:
            w = d.textlength(self.banner, font=self.f_mid)
            d.rounded_rectangle([(self.W - w) / 2 - 20, self.H - 70, (self.W + w) / 2 + 20, self.H - 26], 10,
                                fill=(200, 80, 30, 220))
            d.text(((self.W - w) / 2, self.H - 63), self.banner, font=self.f_mid, fill=(255, 255, 255, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # 右下: カメラ認識結果
        if self.inset is not None:
            ih, iw = 252, 336
            small = cv2.resize(self.inset[40:400, 80:560], (iw, ih))   # 作業領域を切り出して拡大
            x0, y0 = self.W - iw - 18, self.H - ih - 18 - (60 if self.banner else 0)
            out[y0:y0 + ih, x0:x0 + iw] = small
            cv2.rectangle(out, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 2)
            lab = Image.fromarray(cv2.cvtColor(out[y0 - 30:y0, x0:x0 + iw], cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(lab).text((4, 4), "上部カメラの認識結果 (RGB-D)", font=self.f_tiny, fill=(255, 255, 255))
            out[y0 - 30:y0, x0:x0 + iw] = cv2.cvtColor(np.asarray(lab), cv2.COLOR_RGB2BGR)
        return out

    def close(self, speed: float = 1.0):
        self.writer.release()
        vf = f"setpts=PTS/{speed}" if speed != 1.0 else "null"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.tmp_path), "-vf", vf,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", str(self.fps),
                        str(self.out_path)], check=True)
        self.tmp_path.unlink()


# ============================================================================ アプリ本体
class TidyRobotApp:
    def __init__(self, video_path: Path | None, model_path: Path | None = None, layout=None,
                 think_hold: float = 2.5, planner=None):
        self.vc: VideoComposer | None = None
        self.env = TidyEnv(frame_callback=lambda e: self.vc.capture(e) if self.vc else None)
        self.env.reset(layout)
        if video_path:
            self.vc = VideoComposer(self.env, video_path)
        self.perc = Perception(self.env.model)
        self.world = WorldModel()
        self.planner = planner or (LLMPlanner(model_path) if model_path else LLMPlanner())
        if self.vc:
            self.vc.model_label = self.planner.model_name.split("-instruct")[0].replace("qwen", "Qwen")
        self.think_hold = think_hold
        self.log: list[dict] = []
        # 手先の基準 yaw (ホーム姿勢). 立方体を掴むときはこれに最も近い向きを選ぶ
        self.env.kin.qpos[self.env.arm_qadr] = HOME_Q
        mujoco.mj_kinematics(self.env.model, self.env.kin)
        R = self.env.tcp_pose(self.env.kin)[1]
        self.ref_yaw = float(np.arctan2(R[1, 0], R[0, 0]))

    # -------------------------------------------------------------- 観測
    def observe(self, move: bool = True):
        if move:
            self.env.status = "観測姿勢へ (カメラから退避)"
            self.env.move_joints(OBS_Q, 1.2)
            self.env.step(0.2)
        self.env.status = "カメラで認識"
        dets = self.perc.detect(self.env.data)
        self.world.update(dets)
        if self.vc:
            self.vc.inset = self.perc.annotate(list(self.world.dets.values()))
        self.env.step(0.3)
        return self.world

    # -------------------------------------------------------------- 1 つの指示を処理
    def run_command(self, instruction: str) -> dict:
        vc = self.vc
        rec = {"instruction": instruction}
        if vc:
            vc.instruction, vc.reply, vc.interp = instruction, "", ""
            vc.plan_lines, vc.notes, vc.current, vc.done = [], [], -1, set()
        self.observe(move=not np.allclose(self.env.q_cmd, OBS_Q))
        objects = dict(self.world.objects)
        rec["scene"] = objects
        self.env.status = "LLM が考え中…"
        plan = self.planner.plan(instruction, objects)
        rec.update(interpretation=plan.interpretation, reply=plan.reply, raw=plan.raw, steps=plan.steps, notes=plan.rejected, latency=plan.latency)
        print(f"[plan {plan.latency:.1f}s] {instruction} -> {plan.raw}\n  steps={plan.steps} notes={plan.rejected}")
        if vc:
            vc.banner = f"LLM 推論中 (CPU 実時間 {plan.latency:.1f} 秒 → 動画では短縮)"
            vc.hold(self.think_hold)
            vc.banner = ""
            vc.reply = plan.reply
            vc.interp = plan.interpretation
            vc.plan_lines = plan_to_ja(plan.steps, objects)
            vc.notes = [n for n in plan.rejected]
            vc.hold(2.0)
        results = []
        for i, step in enumerate(plan.steps):
            if vc:
                vc.current = i
            ok = self.execute_step(step)
            results.append(ok)
            if vc:
                vc.done.add(i) if ok else None
                vc.current = -1
        if not plan.steps:
            self.env.status = "実行できる手順なし"
            self.env.step(1.5)
        elif not np.allclose(self.env.q_cmd, OBS_Q):
            self.observe()
        self.env.status = ""
        rec["success"] = results
        self.log.append(rec)
        if vc:
            vc.hold(1.0)
        return rec

    # -------------------------------------------------------------- 1 ステップ (ピック&プレース + 確認)
    def execute_step(self, step: dict, max_attempts: int = 2) -> bool:
        pid, qid = step["pick"], step["place"]
        for attempt in range(max_attempts):
            w = self.world if attempt == 0 and np.allclose(self.env.q_cmd, OBS_Q) else self.observe()
            if pid not in w.dets or w.objects[pid].get("hidden"):
                print(f"  ! {pid} を見つけられない / 上に物がある")
                return False
            obj = w.dets[pid]
            height = max(obj.height, 0.02)
            g = min(0.02, height / 2)                      # 上面から把持点までの深さ
            grasp = np.array([obj.xy[0], obj.xy[1], obj.top_z - g])
            if obj.shape == "cube":
                yaw_pick = self.ref_yaw + wrap_quarter(obj.yaw - self.ref_yaw)
            else:
                yaw_pick = self.ref_yaw
            target_xy, support_top, target_yaw = self._place_target(pid, qid, w)
            place = np.array([target_xy[0], target_xy[1], support_top + height - g])
            yaw_place = yaw_pick + (wrap_quarter(target_yaw - obj.yaw) if obj.shape == "cube" else 0.0)
            held = self.env.pick_and_place(grasp, yaw_pick, place, yaw_place)
            # 確認: 再認識して目標位置に置けたか
            w = self.observe()
            d = w.dets.get(pid)
            on = w.objects.get(pid, {}).get("on")
            ok = d is not None and np.linalg.norm(d.xy - target_xy) < 0.03 and (qid == "table" or on == qid)
            print(f"  step {pid}->{qid} attempt{attempt + 1}: held={held} ok={ok} on={on}")
            if ok:
                return True
            self.env.status = "失敗を検知 → 再挑戦"
            self.env.step(0.8)
        return False

    def _place_target(self, pid: str, qid: str, w: WorldModel):
        """置き場所の (xy, 支持面の高さ, 揃えたい yaw) を決める."""
        others = [d.xy for k, d in w.dets.items() if d.kind == "block" and k != pid]
        if qid == "table":
            xs = np.linspace(*TABLE_AREA[0], 6)
            ys = np.linspace(*TABLE_AREA[1], 7)
            cands = [np.array([x, y]) for x in xs for y in ys]
            avoid = others + [d.xy for d in w.dets.values() if d.kind == "plate"]
            best = max(cands, key=lambda c: min(np.linalg.norm(c - o) for o in avoid))
            return best, 0.0, 0.0
        tgt = w.dets[qid]
        if tgt.kind == "block":
            return tgt.xy.copy(), tgt.top_z, tgt.yaw
        # お皿: 3 つの置き場所 (半径 4.2cm, 120° 間隔 → 物体間 7.3cm) のうち最も空いている所
        on_plate = [w.dets[k].xy for k, o in w.objects.items() if o.get("on") == qid and k != pid]
        slots = [tgt.xy + 0.042 * np.array([np.cos(a), np.sin(a)]) for a in np.deg2rad([180, 300, 60])]
        if not on_plate:
            return slots[0], tgt.top_z, 0.0
        best = max(slots, key=lambda c: min(np.linalg.norm(c - o) for o in on_plate))
        return best, tgt.top_z, 0.0

    # -------------------------------------------------------------- 評価用 (シミュレータの正解との比較)
    def perception_error(self) -> dict:
        """認識した位置とシミュレータ内の真の位置との誤差 [mm] (レポートの定量評価用)."""
        errs = {}
        for k, d in self.world.dets.items():
            gt_name = {"red_cube": "red_cube", "yellow_cube": "yellow_cube", "blue_cube": "blue_cube",
                       "green_cylinder": "green_cylinder", "blue_plate": "plate_blue",
                       "pink_plate": "plate_pink"}.get(k)
            if gt_name:
                errs[k] = float(np.linalg.norm(self.env.data.body(gt_name).xpos[:2] - d.xy) * 1000)
        return errs
