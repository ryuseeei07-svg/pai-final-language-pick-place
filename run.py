"""Run one instruction end to end: perceive -> plan with Gemini -> pick & place -> save video.

    python run.py "put the largest red box on the left plate" --out demo.mp4
    python run.py --mock "..."   # canned plan, no API key needed (simulator smoke test)
"""
import argparse
import json

from ppl.geometry import TopCamera
from ppl.sim import BOXES, PLATES, Env


def tagged_image(env, top):
    from PIL import Image, ImageDraw
    img = Image.fromarray(env.top_image())
    d = ImageDraw.Draw(img)
    objects = {}
    for i, (name, xy) in enumerate([(n, tuple(env.box_pos(n)[:2])) for n in BOXES]
                                   + [(n, xy) for n, xy in PLATES.items()]):
        u, v = top.world_to_pixel(*xy)
        d.rectangle([u - 12, v - 12, u + 12, v + 12], fill=(0, 0, 0))
        d.text((u - 6, v - 7), str(i), fill=(255, 255, 255))
        objects[i] = (name, xy)
    import numpy as np
    return np.asarray(img), objects


def make_plan(env, top, instruction, mock):
    if mock:
        plan = json.load(open(mock, encoding="utf-8")) if mock != "builtin" else {"steps": [
            {"pick_label": "red", "pick_point": [0, 0], "place_label": "left plate", "place_point": [0, 0]}]}
        from ppl.vlm import Step
        return [Step(s["pick_label"], tuple(env.box_pos(s["pick_label"])[:2]), s["place_label"],
                     PLATES[s["place_label"]], "mock") for s in plan["steps"]], "mock"
    from ppl.vlm import GeminiPlanner
    planner = GeminiPlanner(top)
    try:
        return planner.plan_points(env.top_image(), instruction), "points"
    except Exception as e:  # malformed JSON, API error, ...
        print("point plan failed, falling back to marks:", e)
        img, objects = tagged_image(env, top)
        return planner.plan_marks(img, instruction, objects), "marks"


def vlm_view(env, top, step, size=250):
    """Top-down image with the pick (red ring) and place (green ring) the plan chose."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(env.top_image())
    d = ImageDraw.Draw(img)
    for xy, col in ((step.pick_xy, (255, 0, 0)), (step.place_xy, (0, 220, 0))):
        u, v = top.world_to_pixel(*xy)
        d.ellipse([u - 22, v - 22, u + 22, v + 22], outline=col, width=5)
    return img.resize((size, size))


def execute(env, top, instruction, mock=None):
    env.rec.caption = f"Instruction: {instruction}"
    env.step(30)
    steps, mode = make_plan(env, top, instruction, mock)
    print(f"plan ({mode}):", steps)
    results = []
    for k, s in enumerate(steps, 1):
        env.rec.caption = f"Instruction: {instruction}\n[{mode}] step {k}/{len(steps)}: {s.pick_label} -> {s.place_label}"
        env.rec.inset = vlm_view(env, top, s)
        got = env.pick(s.pick_xy)
        if got is None and mode == "points":
            # VLM point missed every box: ask again with numbered marks
            from ppl.vlm import GeminiPlanner
            img, objects = tagged_image(env, top)
            s = GeminiPlanner(top).plan_marks(img, f"{instruction} (only: {s.pick_label} -> {s.place_label})", objects)[0]
            env.rec.caption += "\nretry with marks"
            got = env.pick(s.pick_xy)
        if got is None:
            results.append((s.pick_label, "pick failed"))
            continue
        name, grasp = got
        err = env.place(name, s.place_xy)
        results.append((name, grasp, round(err, 3)))
    print("results:", results)
    env.rec.inset = None
    env.rec.caption = "done"
    env.step(60)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="+")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--mock", nargs="?", const="builtin")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--font")
    a = ap.parse_args()
    top = TopCamera()
    env = Env(top, backend="cpu" if a.cpu else "gpu", font_path=a.font)
    for ins in a.instruction:
        execute(env, top, ins, a.mock)
    env.rec.save(a.out)
    print("saved", a.out)
