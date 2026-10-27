"""Check that offscreen rendering works (run this when the video is black)."""
import numpy as np

from ppl.geometry import TopCamera
from ppl.sim import Env

env = Env(TopCamera())
env.step(20)
for name, img in [("top", env.top_image()), ("view", np.asarray(env.view_cam.render()[0]))]:
    print(name, img.shape, img.dtype, "mean", float(np.mean(img)), "max", int(np.max(img)))
