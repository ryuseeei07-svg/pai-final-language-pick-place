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


class _Err(Exception):
    def __init__(self, code):
        self.code = code


def _planner(behaviour, models):
    """GeminiPlanner without a real client: behaviour(model) returns text or raises."""
    import sys
    import types as t
    from ppl.vlm import GeminiPlanner

    class _Models:
        def generate_content(self, model, contents, config):
            return t.SimpleNamespace(text=behaviour(model))

    fake_types = t.SimpleNamespace(
        Part=t.SimpleNamespace(from_bytes=lambda **k: None),
        GenerateContentConfig=lambda **k: None)
    sys.modules["google"] = t.ModuleType("google")
    sys.modules["google.genai"] = t.SimpleNamespace(types=fake_types)
    sys.modules["google.genai.types"] = fake_types
    p = GeminiPlanner.__new__(GeminiPlanner)
    p.client, p.models = t.SimpleNamespace(models=_Models()), models
    return p


def test_falls_back_to_next_model(monkeypatch):
    import numpy as np
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = []

    def behaviour(model):
        calls.append(model)
        if model == "busy":
            raise _Err(503)
        return "ok"

    p = _planner(behaviour, ["busy", "good"])
    import ppl.vlm as v
    monkeypatch.setattr(v, "_png", lambda img: b"")
    assert p._ask(np.zeros((2, 2, 3), dtype=np.uint8), "x") == "ok"
    assert calls == ["busy", "busy", "busy", "good"] and p.last_model == "good"


def test_non_transient_error_is_raised(monkeypatch):
    import numpy as np
    import ppl.vlm as v
    monkeypatch.setattr(v, "_png", lambda img: b"")

    def behaviour(model):
        raise _Err(404)

    p = _planner(behaviour, ["gone", "other"])
    with pytest.raises(_Err):
        p._ask(np.zeros((2, 2, 3), dtype=np.uint8), "x")


def test_step_round_trip():
    from ppl.vlm import Step
    s = Step("red_large", (0.45, 0.12), "left plate", (0.4, 0.27), "r")
    assert Step.from_dict(s.to_dict()) == s


def test_bare_list_is_accepted():
    text = '[{"pick_label": "red", "pick_point": [500, 500], "place_label": "plate", "place_point": [500, 700]}]'
    (step,) = parse_point_plan(text, cam)
    assert step.pick_xy == pytest.approx((0.5, 0.0))
