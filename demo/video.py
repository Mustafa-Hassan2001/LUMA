"""Enhance a video (or the webcam, locally) and write a side-by-side result with an FPS overlay.

    python demo/video.py --checkpoint best.pth --input night.mp4 --output night_enhanced.mp4
    python demo/video.py --checkpoint best.pth --webcam          # local machine only, press q to quit

The FPS overlay shows the model's own throughput (GPU-synchronised), and the
end-of-run summary also reports end-to-end FPS including decoding/encoding.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.app import Enhancer  # noqa: E402


def label(img, text, org=(10, 30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def process(enh: Enhancer, cap, writer=None, show=False, max_frames=None, max_side=None):
    n, model_ms, t0 = 0, [], time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        if max_side and max(frame.shape[:2]) > max_side:
            s = max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out_rgb, ms = enh(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        model_ms.append(ms)
        out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        fps = 1000 / np.mean(model_ms[-30:])
        left, right = frame.copy(), out.copy()
        label(left, "input")
        label(right, f"enhanced | model {fps:.0f} FPS")
        both = np.concatenate([left, right], axis=1)
        if writer is not None:
            writer.write(both)
        if show:
            cv2.imshow("LUMA (q to quit)", both)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    wall = time.perf_counter() - t0
    return {"frames": n, "model_ms_mean": float(np.mean(model_ms)) if model_ms else None,
            "model_fps": float(1000 / np.mean(model_ms)) if model_ms else None,
            "end_to_end_fps": n / wall if wall > 0 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", help="input video file")
    ap.add_argument("--output", help="output .mp4 (side by side)")
    ap.add_argument("--webcam", action="store_true", help="use camera 0 (local use only)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--max_side", type=int, default=1280, help="downscale large frames")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    enh = Enhancer(args.checkpoint, fp16=not args.fp32)
    cap = cv2.VideoCapture(0 if args.webcam else args.input)
    if not cap.isOpened():
        raise SystemExit("could not open " + ("webcam" if args.webcam else str(args.input)))
    writer = None
    if args.output:
        ok, first = cap.read()
        if not ok:
            raise SystemExit("empty video")
        if args.max_side and max(first.shape[:2]) > args.max_side:
            s = args.max_side / max(first.shape[:2])
            first = cv2.resize(first, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = first.shape[:2]
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (2 * w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    stats = process(enh, cap, writer, show=args.webcam, max_frames=args.max_frames,
                    max_side=args.max_side)
    cap.release()
    if writer is not None:
        writer.release()
    if args.webcam:
        cv2.destroyAllWindows()
    print(stats)


if __name__ == "__main__":
    main()
