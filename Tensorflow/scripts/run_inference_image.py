"""Run inference on a single image using an exported TF2 Object Detection SavedModel.

This script is intentionally lightweight and does NOT require the TF Object Detection API
at runtime (only TensorFlow + Pillow + NumPy).

Example:
  .\.venv\Scripts\python.exe Tensorflow\scripts\run_inference_image.py \
    --saved_model_dir Tensorflow\workspace\exported-models\my_ssd_mobnet\saved_model \
    --labels Tensorflow\workspace\annotations\label_map.pbtxt \
    --image Tensorflow\workspace\images\test\example.jpg \
    --output outputs\pred.png

Notes:
- Expects TF2 exporter outputs (exporter_main_v2.py) with keys like:
  detection_boxes, detection_scores, detection_classes, num_detections.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf


def _load_label_map_id_to_name(pbtxt_path: str) -> dict[int, str]:
    """Minimal parser for TF Object Detection API style label_map.pbtxt."""
    text = Path(pbtxt_path).read_text(encoding="utf-8")
    items = re.findall(r"item\s*\{(.*?)\}", text, flags=re.DOTALL)

    id_to_name: dict[int, str] = {}
    for item in items:
        id_match = re.search(r"\bid\s*:\s*(\d+)", item)
        name_match = re.search(r"\bname\s*:\s*'([^']+)'", item) or re.search(
            r'\bname\s*:\s*"([^"]+)"', item
        )
        if not id_match or not name_match:
            continue
        id_to_name[int(id_match.group(1))] = name_match.group(1)

    if not id_to_name:
        raise ValueError(f"No labels parsed from: {pbtxt_path}")
    return id_to_name


def _draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    id_to_name: dict[int, str],
    score_threshold: float,
) -> Image.Image:
    draw = ImageDraw.Draw(image)

    # Optional font; falls back to default.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    width, height = image.size
    for box, cls, score in zip(boxes, classes, scores):
        if float(score) < score_threshold:
            continue

        ymin, xmin, ymax, xmax = [float(v) for v in box]
        left = int(xmin * width)
        right = int(xmax * width)
        top = int(ymin * height)
        bottom = int(ymax * height)

        # Clamp
        left = max(0, min(left, width - 1))
        right = max(0, min(right, width - 1))
        top = max(0, min(top, height - 1))
        bottom = max(0, min(bottom, height - 1))

        label = id_to_name.get(int(cls), str(int(cls)))
        caption = f"{label} {float(score):.2f}"

        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        text_pos = (left + 4, max(0, top - 12))
        draw.text(text_pos, caption, fill=(255, 0, 0), font=font)

    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", required=True, help="Path to exported SavedModel directory")
    parser.add_argument("--labels", required=True, help="Path to label_map.pbtxt")
    parser.add_argument("--image", required=True, help="Path to an input image (jpg/png)")
    parser.add_argument("--output", required=True, help="Path to output image with detections")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Minimum score to draw")

    args = parser.parse_args()

    saved_model_dir = Path(args.saved_model_dir)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel dir not found: {saved_model_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    id_to_name = _load_label_map_id_to_name(args.labels)

    detect_fn = tf.saved_model.load(str(saved_model_dir))

    image = Image.open(image_path).convert("RGB")
    input_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # Standard OD API outputs.
    num = int(detections.get("num_detections")[0])
    boxes = detections["detection_boxes"][0][:num].numpy()
    scores = detections["detection_scores"][0][:num].numpy()

    # Classes are float in many models; convert to int ids.
    classes = detections["detection_classes"][0][:num].numpy().astype(np.int32)

    out_img = image.copy()
    out_img = _draw_boxes(out_img, boxes, classes, scores, id_to_name, args.score_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
