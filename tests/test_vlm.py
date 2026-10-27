import pytest

from ppl.geometry import TopCamera
from ppl.vlm import parse_marks_plan, parse_point_plan

cam = TopCamera()


def test_parse_points_with_code_fence():
    text = '```json\n{"steps": [{"pick_label": "red", "pick_point": [500, 500], "place_label": "plate", "place_point": [500, 700]}]}\n```'
    (step,) = parse_point_plan(text, cam)
    assert step.pick_xy == pytest.approx((0.5, 0.0))
    assert step.place_xy[1] < 0  # right of centre in the image -> -y


def test_parse_marks():
    objs = {0: ("red", (0.5, 0.1)), 1: ("plate", (0.4, -0.2))}
    text = '{"steps": [{"pick_id": 0, "ref_id": 1, "relation": "left_of"}]}'
    (step,) = parse_marks_plan(text, objs)
    assert step.place_xy == pytest.approx((0.4, -0.11))


def test_empty_plan_rejected():
    with pytest.raises(ValueError):
        parse_point_plan('{"steps": []}', cam)
