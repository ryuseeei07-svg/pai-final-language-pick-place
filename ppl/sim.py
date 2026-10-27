"""Genesis scene, Franka controller (IK + joint interpolation) and video recorder."""
import os
import sys

import numpy as np

if sys.platform.startswith("linux"):   # headless servers / Colab need EGL for offscreen rendering
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from .geometry import TopCamera

TABLE_Z = 0.0
BOX_H = 0.04
PLATE_H = 0.01
HAND_OFFSET = 0.11          # hand link height above the box centre when grasping
APPROACH_Z = 0.27
DOWN = np.array([0.0, 1.0, 0.0, 0.0])
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

# name -> (rgb, footprint [m], (x, y)); all boxes share one height so the camera map is exact.
BOXES = {
    "red": ((0.9, 0.1, 0.1), 0.05, (0.45, 0.12)),
    "green": ((0.1, 0.7, 0.2), 0.03, (0.55, 0.00)),
    "blue": ((0.1, 0.2, 0.9), 0.04, (0.45, -0.10)),
    "yellow": ((0.95, 0.85, 0.1), 0.05, (0.62, 0.17)),
    "purple": ((0.6, 0.2, 0.8), 0.03, (0.62, -0.17)),
}
PLATES = {
    "left plate": (0.40, 0.27),
    "right plate": (0.40, -0.27),
}


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class Recorder:
    """Collects frames from a viewing camera and burns a caption into them."""

    def __init__(self, cam, every=3, font_path=None):
        self.cam, self.every, self.font_path = cam, every, font_path
        self.frames, self.caption, self._n = [], "", 0
        self.inset = None   # PIL image shown in the bottom-right corner (what the VLM sees)

    def tick(self):
        self._n += 1
        if self._n % self.every == 0:
            self.frames.append(self._decorate(_np(self.cam.render()[0]).astype(np.uint8)))

    def _decorate(self, frame):
        if not self.caption and self.inset is None:
            return frame
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        if self.inset is not None:
            img.paste(self.inset, (img.width - self.inset.width - 8, img.height - self.inset.height - 8))
        try:
            font = ImageFont.truetype(self.font_path, 22) if self.font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        d = ImageDraw.Draw(img)
        if not self.caption:
            return np.asarray(img)
        d.rectangle([0, 0, img.width, 64], fill=(0, 0, 0))
        d.multiline_text((10, 8), self.caption, fill=(255, 255, 255), font=font)
        return np.asarray(img)

    def save(self, path, fps=30):
        import imageio
        if self.frames:
            mean = float(np.mean(self.frames[len(self.frames) // 2]))
            print(f"frames: {len(self.frames)}, mid-frame mean brightness: {mean:.1f}")
            if mean < 5:
                print("WARNING: frames are almost black -> offscreen rendering is not working (see notebook cell 'debug render').")
        try:
            imageio.mimsave(path, self.frames, fps=fps)
        except OSError as e:  # ffmpeg could not start: keep the frames instead of losing the run
            import os
            d = os.path.splitext(path)[0] + "_frames"
            os.makedirs(d, exist_ok=True)
            for i, f in enumerate(self.frames):
                imageio.imwrite(os.path.join(d, f"{i:05d}.png"), f)
            print(f"video write failed ({e}); wrote PNG frames to {d}")


class Env:
    def __init__(self, top: TopCamera, show_viewer=False, backend="gpu", font_path=None):
        import genesis as gs
        self.gs = gs
        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu, logging_level="warning")
        self.top = top
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=show_viewer)
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.78, 0.72, 0.62)))
        self.plates = {
            n: self.scene.add_entity(
                gs.morphs.Box(size=(0.14, 0.14, PLATE_H), pos=(x, y, PLATE_H / 2), fixed=True),
                surface=gs.surfaces.Default(color=(0.95, 0.95, 0.95)))
            for n, (x, y) in PLATES.items()
        }
        self.boxes = {
            n: self.scene.add_entity(
                gs.morphs.Box(size=(f, f, BOX_H), pos=(x, y, BOX_H / 2)),
                material=gs.materials.Rigid(friction=1.5),
                surface=gs.surfaces.Default(color=c))
            for n, (c, f, (x, y)) in BOXES.items()
        }
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.top_cam = self.scene.add_camera(
            res=(top.res, top.res), pos=(top.cx, top.cy, top.height),
            lookat=(top.cx, top.cy, 0.0), up=(1.0, 0.0, 0.0), fov=top.fov_deg, GUI=False)
        self.view_cam = self.scene.add_camera(
            res=(960, 540), pos=(1.3, -0.85, 0.85), lookat=(0.45, 0.0, 0.05), fov=45, GUI=False)
        self.scene.build()
        self.rec = Recorder(self.view_cam, font_path=font_path)
        self.motors = np.arange(7)
        self.fingers = np.arange(7, 9)
        self.hand = self.franka.get_link("hand")
        self.attached = None   # (box entity, offset) while the weld-like fallback is active
        f = self.franka
        f.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        f.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        f.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                               np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
        self.reset_arm()

    # --- low level -------------------------------------------------------
    def step(self, n=1):
        for _ in range(n):
            if self.attached is not None:
                box, off = self.attached
                box.set_pos(_np(self.hand.get_pos()) + off)
            self.scene.step()
            self.rec.tick()

    def reset_arm(self):
        self.franka.set_qpos(HOME)
        self.franka.control_dofs_position(HOME)
        self.step(60)

    def top_image(self):
        return _np(self.top_cam.render()[0]).astype(np.uint8)

    def box_pos(self, name):
        return _np(self.boxes[name].get_pos())

    def nearest_box(self, xy):
        d = {n: np.linalg.norm(self.box_pos(n)[:2] - np.asarray(xy)) for n in self.boxes}
        n = min(d, key=d.get)
        return n, d[n]

    # --- motion ----------------------------------------------------------
    def move_hand(self, xyz, steps=80):
        q_goal = _np(self.franka.inverse_kinematics(
            link=self.hand, pos=np.asarray(xyz, dtype=float), quat=DOWN,
            init_qpos=self.franka.get_qpos()))
        q0 = _np(self.franka.get_qpos())
        for i in range(1, steps + 1):
            q = q0 + (q_goal - q0) * i / steps
            self.franka.control_dofs_position(q[:7], self.motors)
            self.step()

    def go_home(self, steps=100):
        q0 = _np(self.franka.get_qpos())
        for i in range(1, steps + 1):
            q = q0 + (HOME - q0) * i / steps
            self.franka.control_dofs_position(q[:7], self.motors)
            self.step()

    def open_gripper(self, steps=40):
        self.attached = None
        self.franka.control_dofs_position(np.array([0.04, 0.04]), self.fingers)
        self.step(steps)

    def close_gripper(self, steps=60):
        self.franka.control_dofs_force(np.array([-1.0, -1.0]), self.fingers)
        self.step(steps)

    # --- task ------------------------------------------------------------
    def support_height(self, xy):
        """Height of the surface at xy (plate top if on a plate, else the table)."""
        for n, (px, py) in PLATES.items():
            if abs(xy[0] - px) < 0.075 and abs(xy[1] - py) < 0.075:
                return PLATE_H
        return TABLE_Z

    def pick(self, xy, allow_attach=True):
        """Returns (box name, mode) where mode is 'physical' or 'attached'; None if the miss is hopeless."""
        name, dist = self.nearest_box(xy)
        if dist > 0.05:
            return None
        x, y = self.box_pos(name)[:2]       # snap to the box under the VLM point
        gz = BOX_H / 2 + HAND_OFFSET
        self.open_gripper()
        self.move_hand((x, y, APPROACH_Z))
        self.move_hand((x, y, gz))
        self.close_gripper()
        self.move_hand((x, y, APPROACH_Z), steps=80)
        if self.box_pos(name)[2] > 0.08:
            return name, "physical"
        if not allow_attach:
            return None
        # fallback: re-grasp and carry the box with the hand
        self.open_gripper()
        self.move_hand((x, y, gz))
        self.close_gripper(30)
        off = self.box_pos(name) - _np(self.hand.get_pos())
        self.attached = (self.boxes[name], off)
        self.move_hand((x, y, APPROACH_Z), steps=80)
        return name, "attached"

    def place(self, name, xy):
        z = self.support_height(xy) + BOX_H / 2 + HAND_OFFSET + 0.005
        self.move_hand((xy[0], xy[1], APPROACH_Z), steps=100)
        self.move_hand((xy[0], xy[1], z))
        self.open_gripper()
        self.move_hand((xy[0], xy[1], APPROACH_Z), steps=60)
        self.go_home()
        return float(np.linalg.norm(self.box_pos(name)[:2] - np.asarray(xy)))
