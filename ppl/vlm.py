"""VLM planner: turns (top-down image, instruction) into pick/place steps.

Primary mode asks Gemini for image points; fallback mode draws numbered marks on the
objects and asks Gemini to pick marks plus a spatial relation (set-of-marks).
"""
import io
import json
import os
from dataclasses import dataclass

from .geometry import TopCamera, relation_to_world

DEFAULT_MODEL = "gemini-2.5-flash"

POINT_PROMPT = """You control a robot arm above a table. The image is a top-down view; the robot base is at the bottom edge.
Image left/right/up/down are what you see in the image.
Instruction: "{instruction}"

Break the instruction into pick-and-place steps (one box per step, in the order they should be done).
For each step give the box to pick and the place target as image points.
Answer with JSON only:
{{"steps": [{{"pick_label": "<short description>", "pick_point": [y, x],
 "place_label": "<short description>", "place_point": [y, x], "reason": "<one short sentence>"}}]}}
Points are [y, x] normalised to 0-1000. A place target on a plate means the plate centre;
a place target next to a box means a free table spot about 9 cm (about 70 px) away from that box in the image."""

MARKS_PROMPT = """You control a robot arm above a table. The image is a top-down view; the robot base is at the bottom edge.
Objects carry white number tags. Instruction: "{instruction}"

Break the instruction into pick-and-place steps. For each step give the tag number of the box to pick,
the tag number of the reference object for the place target, and the relation of the place spot to it:
one of on, above, below, left_of, right_of (as seen in the image).
Answer with JSON only:
{{"steps": [{{"pick_id": 0, "ref_id": 1, "relation": "on", "reason": "<one short sentence>"}}]}}"""


@dataclass
class Step:
    pick_label: str
    pick_xy: tuple[float, float]
    place_label: str
    place_xy: tuple[float, float]
    reason: str = ""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def parse_point_plan(text: str, cam: TopCamera) -> list[Step]:
    steps = []
    for s in _extract_json(text)["steps"]:
        steps.append(Step(
            s.get("pick_label", "box"), cam.gemini_to_world(*s["pick_point"]),
            s.get("place_label", "target"), cam.gemini_to_world(*s["place_point"]),
            s.get("reason", ""),
        ))
    if not steps:
        raise ValueError("empty plan")
    return steps


def parse_marks_plan(text: str, objects: dict[int, tuple[str, tuple[float, float]]]) -> list[Step]:
    """objects: tag id -> (label, world xy)."""
    steps = []
    for s in _extract_json(text)["steps"]:
        pick_label, pick_xy = objects[int(s["pick_id"])]
        ref_label, ref_xy = objects[int(s["ref_id"])]
        steps.append(Step(pick_label, pick_xy, f"{s['relation']} {ref_label}",
                          relation_to_world(ref_xy, s["relation"]), s.get("reason", "")))
    if not steps:
        raise ValueError("empty plan")
    return steps


def _png(image) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


class GeminiPlanner:
    def __init__(self, cam: TopCamera, model: str | None = None):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (Colab secret or environment variable).")
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.cam = cam

    def _ask(self, image, prompt: str) -> str:
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=_png(image), mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
        )
        return resp.text

    def plan_points(self, image, instruction: str) -> list[Step]:
        return parse_point_plan(self._ask(image, POINT_PROMPT.format(instruction=instruction)), self.cam)

    def plan_marks(self, tagged_image, instruction: str, objects) -> list[Step]:
        return parse_marks_plan(self._ask(tagged_image, MARKS_PROMPT.format(instruction=instruction)), objects)
