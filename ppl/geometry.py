"""Pixel <-> world conversion for the fixed top-down camera.

The camera looks straight down with image-up = world +x, so image-right = world -y
and the robot base sits at the bottom of the image. All boxes share one top height
(plane_z), so the mapping is an exact similarity transform on that plane.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TopCamera:
    cx: float = 0.5          # world x of the image centre
    cy: float = 0.0          # world y of the image centre
    height: float = 1.5      # camera height above the table
    fov_deg: float = 30.0    # vertical field of view
    res: int = 640           # square image
    plane_z: float = 0.04    # height of the box tops

    @property
    def metres_per_pixel(self) -> float:
        d = self.height - self.plane_z
        return 2.0 * d * math.tan(math.radians(self.fov_deg) / 2.0) / self.res

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        s = self.metres_per_pixel
        return self.cx - (v - self.res / 2) * s, self.cy - (u - self.res / 2) * s

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        s = self.metres_per_pixel
        return (self.cy - y) / s + self.res / 2, (self.cx - x) / s + self.res / 2

    def gemini_to_world(self, y1000: float, x1000: float) -> tuple[float, float]:
        """Gemini returns points as [y, x] normalised to 0..1000."""
        return self.pixel_to_world(x1000 / 1000 * self.res, y1000 / 1000 * self.res)


# Image-frame relation -> world offset (metres). Image up = +x, image right = -y.
RELATION_OFFSETS = {
    "on": (0.0, 0.0),
    "above": (0.09, 0.0),
    "below": (-0.09, 0.0),
    "left_of": (0.0, 0.09),
    "right_of": (0.0, -0.09),
}


def relation_to_world(ref_xy: tuple[float, float], relation: str) -> tuple[float, float]:
    if relation not in RELATION_OFFSETS:
        raise ValueError(f"unknown relation: {relation}")
    dx, dy = RELATION_OFFSETS[relation]
    return ref_xy[0] + dx, ref_xy[1] + dy
