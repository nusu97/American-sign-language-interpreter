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

## Example Classes

This repo's current label map includes example gestures such as:

- Hello
- yes
- No
- thank you
- I love you

Update `Tensorflow/workspace/annotations/label_map.pbtxt` to match your dataset.

## Demo (What to Show)

- Run inference on a single image and produce `outputs/prediction.png` with bounding boxes + labels.
- Export an inference artifact (SavedModel) for deployment workflows.

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

## Getting Started

### 1) Setup

From the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-venv.txt
```

Compile the Object Detection API protos (one-time):

```powershell
.\.venv\Scripts\python.exe -m grpc_tools.protoc `
  -I Tensorflow\models\research `
  --python_out=Tensorflow\models\research `
  --grpc_python_out=Tensorflow\models\research `
  Tensorflow\models\research\object_detection\protos\*.proto
```

### 2) Data

Put images here:

- `Tensorflow/workspace/images/train/`
- `Tensorflow/workspace/images/test/`

Label images (Pascal VOC XML) using LabelImg:

```powershell
.\.venv\Scripts\python.exe Tensorflow\labelImg\labelImg.py
```

Update labels in:

- `Tensorflow/workspace/annotations/label_map.pbtxt`

Generate TFRecords:

```powershell
.\.venv\Scripts\python.exe Tensorflow\scripts\generate_tfrecord.py `
  -x Tensorflow\workspace\images\train `
  -i Tensorflow\workspace\images\train `
  -l Tensorflow\workspace\annotations\label_map.pbtxt `
  -o Tensorflow\workspace\annotations\train.record

.\.venv\Scripts\python.exe Tensorflow\scripts\generate_tfrecord.py `
  -x Tensorflow\workspace\images\test `
  -i Tensorflow\workspace\images\test `
  -l Tensorflow\workspace\annotations\label_map.pbtxt `
  -o Tensorflow\workspace\annotations\test.record
```

### 3) Configure training

Edit:

- `Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config`

Make sure these are correct:

- `num_classes` matches your label map
- `train_input_reader.input_path` and `eval_input_reader.input_path` point to `train.record` / `test.record`
- `label_map_path` points to `label_map.pbtxt`
- `fine_tune_checkpoint` points to your pre-trained checkpoint (typically `.../ckpt-0`)

If training diverges (loss becomes huge / `inf` / `nan`), use a conservative LR+warmed cosine schedule:

```text
learning_rate_base: 0.01
warmup_learning_rate: 0.0025
total_steps: 10000
warmup_steps: 1000
```

### 4) Train

This script sets `PYTHONPATH` correctly for the vendored OD-API:

```powershell
.\run_train.ps1 -NumTrainSteps 10000 -AppendLog
```

Outputs go to:

- `Tensorflow/workspace/models/my_ssd_mobnet/`

### 5) Export (SavedModel)

```powershell
$researchDir = Resolve-Path "Tensorflow\models\research"
$modelsRoot  = Resolve-Path "Tensorflow\models"
$env:PYTHONPATH = "$researchDir;$researchDir\slim;$modelsRoot"

Push-Location $researchDir
try {
	..\..\..\.venv\Scripts\python.exe object_detection\exporter_main_v2.py `
	  --input_type image_tensor `
	  --pipeline_config_path "..\..\..\Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config" `
	  --trained_checkpoint_dir "..\..\..\Tensorflow\workspace\models\my_ssd_mobnet" `
	  --output_directory "..\..\..\Tensorflow\workspace\exported-models\my_ssd_mobnet"
}
finally {
	Pop-Location
}
```

### 6) Inference (single image)

SavedModel inference:

```powershell
.\.venv\Scripts\python.exe Tensorflow\scripts\run_inference_image.py `
  --saved_model_dir Tensorflow\workspace\exported-models\my_ssd_mobnet\saved_model `
  --labels Tensorflow\workspace\annotations\label_map.pbtxt `
  --image Tensorflow\workspace\images\test\your_image.jpg `
  --output outputs\prediction.png
```

If SavedModel loading fails on your machine, export a frozen graph and use the fallback script:

```powershell
$repo = Resolve-Path "."
$researchDir = Resolve-Path "Tensorflow\models\research"
$modelsRoot  = Resolve-Path "Tensorflow\models"
$env:PYTHONPATH = "$researchDir;$researchDir\slim;$modelsRoot"

.\.venv\Scripts\python.exe "$researchDir\object_detection\export_inference_graph.py" `
  --input_type image_tensor `
  --pipeline_config_path Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config `
  --trained_checkpoint_prefix Tensorflow\workspace\models\my_ssd_mobnet\ckpt-<N> `
  --output_directory Tensorflow\workspace\exported-graph\my_ssd_mobnet

.\.venv\Scripts\python.exe Tensorflow\scripts\run_inference_frozen_graph.py `
  --frozen_graph Tensorflow\workspace\exported-graph\my_ssd_mobnet\frozen_inference_graph.pb `
  --labels Tensorflow\workspace\annotations\label_map.pbtxt `
  --image Tensorflow\workspace\images\test\your_image.jpg `
  --output outputs\prediction.png
```

## Troubleshooting

- Protobuf errors: run everything from this repo's `.venv` (this repo pins `protobuf<5`).
- `inf`/`nan` loss: lower LR, increase warmup, verify `num_classes` and label map IDs, regenerate TFRecords.

## License

See [LICENSE](LICENSE).