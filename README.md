# American Sign Language Interpreter (TF Object Detection)

An end-to-end computer vision project that trains a custom TensorFlow Object Detection model (SSD MobileNet) to recognize American Sign Language (ASL) gestures from images.

This repo vendors the TensorFlow Object Detection API under `Tensorflow/models` and includes a Windows-friendly workflow: label data → TFRecords → train → export → run inference.

## Project Overview

- **Problem:** Recognize ASL gestures with a lightweight detector that can be deployed for real-time usage.
- **Approach:** Fine-tune an SSD MobileNet model with the TensorFlow Object Detection API using custom labeled images.
- **Output:** Exportable model (SavedModel) + scripts to run inference on a single image and save an annotated prediction.

## Highlights

- Built a reproducible training pipeline using `pipeline.config` + TFRecords.
- Implemented dataset tooling (Pascal VOC XML → TFRecord) with `Tensorflow/scripts/generate_tfrecord.py`.
- Added inference utilities (`Tensorflow/scripts/run_inference_image.py` and a frozen-graph fallback) for quick demo generation.

## Documentation

- Confluence: https://nusu97.atlassian.net/wiki/spaces/~71202024adedf74f4d48b6bb3ef03154dffd28/pages/196813/Documentation+for+ASL+Detection

## Example Classes

This repo's current label map includes example gestures such as:

- Hello
- yes
- No
- thank you
- I love you

Update `Tensorflow/workspace/annotations/label_map.pbtxt` to match your dataset.

## Tech Stack

- Python 3.11
- TensorFlow 2.x
- TensorFlow Models (Object Detection API)
- Protobuf configs + TFRecord
- LabelImg (Pascal VOC annotations)

## Repo Structure (high level)

- `Tensorflow/workspace/` — dataset, annotations, models, exported artifacts
- `Tensorflow/scripts/` — TFRecord generation + inference scripts
- `run_train.ps1` — training helper (sets `PYTHONPATH` correctly for the vendored OD-API)

## Requirements

- Windows + PowerShell
- Python 3.11
- A GPU is optional (CPU works, slower)

See [LICENSE](LICENSE).