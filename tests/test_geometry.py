import math

import pytest

from ppl.geometry import TopCamera, relation_to_world

cam = TopCamera()


def test_round_trip():
    for x, y in [(0.5, 0.0), (0.4, 0.2), (0.65, -0.3)]:
        u, v = cam.world_to_pixel(x, y)
        assert cam.pixel_to_world(u, v) == pytest.approx((x, y), abs=1e-9)


def test_orientation():
    u0, v0 = cam.world_to_pixel(0.5, 0.0)
    assert (u0, v0) == (320, 320)
    _, v_far = cam.world_to_pixel(0.7, 0.0)   # farther from robot -> higher in image
    assert v_far < v0
    u_neg_y, _ = cam.world_to_pixel(0.5, -0.1)  # -y -> right in image
    assert u_neg_y > u0


def test_gemini_centre():
    x, y = cam.gemini_to_world(500, 500)
    assert math.isclose(x, 0.5, abs_tol=1e-9) and math.isclose(y, 0.0, abs_tol=1e-9)


def test_relation():
    assert relation_to_world((0.5, 0.0), "left_of") == (0.5, 0.09)
