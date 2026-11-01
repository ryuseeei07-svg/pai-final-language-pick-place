"""Join the official OP + demo + ED into one 1080p video (no audio).

    python make_final_video.py demo.mp4 OP.mp4 ED.mp4 final.mp4
The official OP/ED files are not part of this repository.
"""
import subprocess
import sys

import imageio_ffmpeg

demo, op, ed, out = sys.argv[1:5]
W, H, FPS = 1920, 1080, 30
norm = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p"
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", op, "-i", demo, "-i", ed,
    "-filter_complex", f"[0:v]{norm}[a];[1:v]{norm}[b];[2:v]{norm}[c];[a][b][c]concat=n=3:v=1:a=0[v]",
    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out,
]
subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
print("wrote", out)
