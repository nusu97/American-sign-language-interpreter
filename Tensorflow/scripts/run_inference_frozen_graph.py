"""Run inference on a single image using a frozen inference graph (.pb).

This is a compatibility path for Windows/TF OD-API setups where
`tf.saved_model.load()` may fail on exported SavedModels.

Requires:
- TensorFlow (uses tf.compat.v1)
- Pillow, NumPy

Example:
  .\.venv\Scripts\python.exe Tensorflow\scripts\run_inference_frozen_graph.py \
    --frozen_graph Tensorflow\workspace\exported-graph\my_ssd_mobnet\frozen_inference_graph.pb \
    --labels Tensorflow\workspace\annotations\label_map.pbtxt \
    --image Tensorflow\workspace\images\test\example.jpg \
    --output outputs\pred.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow.compat.v1 as tf


def _load_label_map_id_to_name(pbtxt_path: str) -> dict[int, str]:
    text = Path(pbtxt_path).read_text(encoding="utf-8")
    items = re.findall(r"item\s*\{(.*?)\}", text, flags=re.DOTALL)
    id_to_name: dict[int, str] = {}
    for item in items:
        id_match = re.search(r"\bid\s*:\s*(\d+)", item)
        name_match = re.search(r"\bname\s*:\s*'([^']+)'", item) or re.search(
            r'\bname\s*:\s*"([^"]+)"', item
        )
        if id_match and name_match:
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
        left = max(0, min(left, width - 1))
        right = max(0, min(right, width - 1))
        top = max(0, min(top, height - 1))
        bottom = max(0, min(bottom, height - 1))

        label = id_to_name.get(int(cls), str(int(cls)))
        caption = f"{label} {float(score):.2f}"
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        draw.text((left + 4, max(0, top - 12)), caption, fill=(255, 0, 0), font=font)

    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph", required=True, help="Path to frozen_inference_graph.pb")
    parser.add_argument("--labels", required=True, help="Path to label_map.pbtxt")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--score_threshold", type=float, default=0.3)

    args = parser.parse_args()

    frozen_graph = Path(args.frozen_graph)
    if not frozen_graph.exists():
        raise FileNotFoundError(f"Frozen graph not found: {frozen_graph}")

    id_to_name = _load_label_map_id_to_name(args.labels)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(frozen_graph.read_bytes())
        tf.import_graph_def(graph_def, name="")

    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
    boxes_tensor = detection_graph.get_tensor_by_name("detection_boxes:0")
    scores_tensor = detection_graph.get_tensor_by_name("detection_scores:0")
    classes_tensor = detection_graph.get_tensor_by_name("detection_classes:0")
    num_tensor = detection_graph.get_tensor_by_name("num_detections:0")

    with tf.Session(graph=detection_graph) as sess:
        boxes, scores, classes, num = sess.run(
            [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
            feed_dict={image_tensor: image_np[None, ...]},
        )

    n = int(num[0])
    boxes = boxes[0][:n]
    scores = scores[0][:n]
    classes = classes[0][:n].astype(np.int32)

    out_img = _draw_boxes(image.copy(), boxes, classes, scores, id_to_name, args.score_threshold)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
