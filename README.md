# American Sign Language Interpreter (TensorFlow Object Detection)

Train a custom TensorFlow Object Detection model (SSD MobileNet) to recognize American Sign Language (ASL) gestures from images/video.

This repo vendors the TensorFlow Models / Object Detection API under `Tensorflow/models` and provides a Windows-friendly workflow:

- Label images with LabelImg
- Convert annotations (Pascal VOC XML) → TFRecord
- Train with a pre-trained SSD MobileNet checkpoint using a reproducible `pipeline.config`

## Tech Stack

- Python 3.11 (tested on Windows)
- TensorFlow 2.x
- TensorFlow Object Detection API (from `Tensorflow/models`)
- TFRecord + Protobuf configs (`pipeline.config`)
- LabelImg (for annotation)

## Repository Layout

- `Tensorflow/models/` — TensorFlow Models repo (includes Object Detection API)
- `Tensorflow/labelImg/` — LabelImg annotation tool
- `Tensorflow/scripts/generate_tfrecord.py` — converts LabelImg XML → TFRecord
- `Tensorflow/workspace/`
	- `images/train/` and `images/test/` — your raw images
	- `annotations/` — `label_map.pbtxt`, `train.record`, `test.record`
	- `pre-trained-models/` — downloaded pre-trained checkpoints
	- `models/my_ssd_mobnet/` — your training outputs + `pipeline.config`
- `run_train.ps1` — PowerShell helper to launch training with correct `PYTHONPATH`

## Setup (Windows)

### 1) Create a virtual environment

From the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install Python dependencies

```powershell
pip install -r requirements-venv.txt
```

`requirements-venv.txt` pins `protobuf<5` to avoid common TensorFlow/OD-API compatibility issues.

### 3) Compile Object Detection API protos (required)

The Object Detection API uses `.proto` files that must be compiled once.

Run this from the repo root:

```powershell
.\.venv\Scripts\python.exe -m grpc_tools.protoc `
	-I Tensorflow\models\research `
	--python_out=Tensorflow\models\research `
	--grpc_python_out=Tensorflow\models\research `
	Tensorflow\models\research\object_detection\protos\*.proto
```

If you see errors about missing `grpc_tools`, ensure `grpcio-tools` is installed (it is listed in `requirements-venv.txt`).

## Data Preparation

### 1) Collect images

Put images here:

- `Tensorflow/workspace/images/train/`
- `Tensorflow/workspace/images/test/`

Recommended: keep consistent lighting/backgrounds at first, then diversify.

### 2) Label images with LabelImg

LabelImg is vendored under `Tensorflow/labelImg`.

Typical flow:

1. Open LabelImg
2. Set the image directory to `.../Tensorflow/workspace/images/train` (and later `test`)
3. Save annotations as Pascal VOC XML into the same folder as the images

Launching (one option):

```powershell
.\.venv\Scripts\python.exe Tensorflow\labelImg\labelImg.py
```

Note: LabelImg may require additional GUI dependencies depending on your machine.

### 3) Update the label map

The label map is at `Tensorflow/workspace/annotations/label_map.pbtxt`.

Current example labels in this repo:

- Hello
- yes
- No
- thank you
- I love you
- ww

Keep label names consistent between LabelImg and `label_map.pbtxt`.

### 4) Generate TFRecords

This converts the LabelImg XML files into `.record` files used by training.

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

The generator includes guardrails for common protobuf/TensorFlow interpreter mismatches and will warn if you’re not using the repo `.venv`.

## Pre-trained Model

This project is configured to fine-tune from a pre-trained SSD MobileNet checkpoint (COCO).

Place the extracted model checkpoint folder under:

`Tensorflow/workspace/pre-trained-models/`

Then verify `fine_tune_checkpoint` inside your `pipeline.config` points at `.../checkpoint/ckpt-0`.

## Training

### 1) Configure the pipeline

Your training config is:

`Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config`

Key fields you’ll typically edit:

- `num_classes` (must match your label map)
- `train_input_reader.input_path` / `eval_input_reader.input_path`
- `label_map_path`
- `fine_tune_checkpoint`
- optimizer learning rate schedule (important for stability)

#### Stable cosine-decay values (recommended)

If you encounter exploding loss / `inf` / `nan`, reduce the learning rate and use a longer warmup. A stable baseline is:

```text
learning_rate_base: 0.01
warmup_learning_rate: 0.0025
total_steps: 10000
warmup_steps: 1000
```

### 2) Start training (PowerShell)

Use the helper script which sets `PYTHONPATH` correctly for the vendored Object Detection API:

```powershell
.\run_train.ps1 -NumTrainSteps 10000 -AppendLog
```

Useful flags:

- `-KillExisting` — stops a previous run targeting the same model dir
- `-NoWait` — detaches and returns immediately
- `-AppendLog` — writes a timestamped log file

Training outputs go to:

`Tensorflow/workspace/models/my_ssd_mobnet/`

### 3) Monitor training

If TensorBoard is available in your environment:

```powershell
python -m tensorboard.main --logdir Tensorflow\workspace\models\my_ssd_mobnet
```

## Demo / Results

This repo includes example training logs you can use as a quick “sanity check” for stability.

### What “good” looks like

Early in training, `Loss/total_loss` should be a small, finite value (often in the single digits) and should not jump to extremely large numbers.

Example stable run (Step 100):

- Total loss stays finite (~1.28): [train_smoke_20260111_144608.log.err](train_smoke_20260111_144608.log.err#L97-L107)

### What “bad” looks like

If the run diverges, losses can explode to huge values and often lead to `inf`/`nan` checkpoints.

Example diverged run (Step 100):

- Total loss explodes (~$3.09\times10^{18}$): [train_smoke.log.err](train_smoke.log.err#L97-L107)

### Reproducing the sanity check

Run a short training burst and confirm the losses are finite:

```powershell
.\run_train.ps1 -NumTrainSteps 100 -AppendLog
```

Then inspect the generated `*.log.err` file for the Step 100 loss dictionary.

## Notes on Metrics (mAP)

Mean Average Precision (mAP) is produced during evaluation, not during training-only runs. If you want mAP reported in the logs, make sure evaluation is enabled and run the eval job against your trained checkpoints.

## Resetting Corrupted Checkpoints

If your checkpoints are corrupted (e.g., contain `inf` values), delete everything in the model directory **except** `pipeline.config`, then restart training.

PowerShell (from repo root):

```powershell
$d = "Tensorflow\workspace\models\my_ssd_mobnet"
Get-ChildItem $d -Force | Where-Object { $_.Name -ne "pipeline.config" } | Remove-Item -Recurse -Force
```

## Export + Inference Quickstart

Once you have a trained checkpoint in `Tensorflow/workspace/models/my_ssd_mobnet/`, export a TF2 SavedModel and run inference on a single image.

### 1) Export a SavedModel

From the repo root:

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

Output:

- `Tensorflow/workspace/exported-models/my_ssd_mobnet/saved_model/`
- a copy of your `pipeline.config` inside the export folder

### 2) Run inference on a single image

Pick a test image (any `.jpg`/`.png`) and run:

```powershell
.\.venv\Scripts\python.exe Tensorflow\scripts\run_inference_image.py `
	--saved_model_dir Tensorflow\workspace\exported-models\my_ssd_mobnet\saved_model `
	--labels Tensorflow\workspace\annotations\label_map.pbtxt `
	--image Tensorflow\workspace\images\test\your_image.jpg `
	--output outputs\prediction.png
```

This writes an annotated image to `outputs/prediction.png`.

## Troubleshooting

### Protobuf / TensorFlow import errors

Symptoms:

- `ImportError: cannot import name 'runtime_version' from google.protobuf ...`

Fix:

- Ensure you are running **everything** from the repo `.venv`
- Keep `protobuf<5` (already pinned)

### Training loss becomes `inf`/`nan`

Common causes:

- learning rate too high
- warmup too short
- label map mismatch (wrong `num_classes`, missing label IDs)
- bad annotations (empty boxes, invalid coordinates)

Fixes:

- Use the stable cosine-decay values above
- Double-check `label_map.pbtxt` and `num_classes`
- Regenerate TFRecords after labeling changes

### Training won’t start / can’t import `object_detection`

Use `run_train.ps1` (it sets `PYTHONPATH` to the vendored `Tensorflow/models` tree). Running `model_main_tf2.py` directly without `PYTHONPATH` is a common cause.

## License

See [LICENSE](LICENSE).