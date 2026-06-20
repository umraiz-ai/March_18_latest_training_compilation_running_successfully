# YOLOv9 March — Training and TACHY Compilation Guide

This repository is a March 2025 workspace for training BSNet models from scratch and deploying them through the **TACHY-RT** compilation pipeline recommended by Deeper-I. The workflow produces two PyTorch checkpoints—a base model and an optimized variant—which are then merged, quantized, and packaged for hardware runtime.

---

## Environment Setup

### Conda environment

Create and activate the dedicated environment:

```bash
conda activate umz_second
```

All Python dependencies are pinned in [`requirements_umz_second.txt`](requirements_umz_second.txt). Install them after cloning:

```bash
pip install -r requirements_umz_second.txt
```

> **Note:** A missing TensorFlow import warning during startup is expected and does not block training.

---

## Training Overview

Training runs in **two stages**:

| Stage | Config | Purpose | Output |
|-------|--------|---------|--------|
| 1 | `models/deeper-i/bsnet-t.yaml` | Train from scratch | Base `.pt` checkpoint |
| 2 | `models/deeper-i/bsnet-t-o.yaml` | Fine-tune with XWN optimization settings | Optimized `.pt` checkpoint |

Both stages use distributed training on four GPUs via `torchrun`. Adjust dataset paths, run names, and weight paths to match your environment.

### Stage 1 — Scratch training

```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py \
  --workers 16 \
  --device 0,1,2,3 \
  --batch-size 64 \
  --data /srv/DATA/DATASETS/NIPA_Data_2025_v9/data.yaml \
  --img 416 \
  --cfg models/deeper-i/bsnet-t.yaml \
  --weights '' \
  --name bsnet-t \
  --hyp data/hyps/hyp.scratch.yaml \
  --min-items 0 \
  --epochs 300 \
  --close-mosaic 3 \
  --optimizer SGD
```

Checkpoints are written under `runs/train/<name>/weights/` (e.g. `best.pt`, `last.pt`).

### Stage 2 — Optimization training

Load the Stage 1 `best.pt` and continue with the optimization config:

```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py \
  --workers 16 \
  --device 0,1,2,3 \
  --batch-size 64 \
  --data /srv/DATA/DATASETS/NIPA_Data_2025_v9/data.yaml \
  --img 416 \
  --cfg models/deeper-i/bsnet-t-o.yaml \
  --weights /home/contil/umraiz/yolov9_march/runs/train/bsnet-t/weights/best.pt \
  --name bsnet-t-o \
  --hyp data/hyps/hyp.optimize.yaml \
  --min-items 0 \
  --epochs 150 \
  --close-mosaic 3 \
  --optimizer SGD
```

Update `--weights` to point at your Stage 1 output before running.

---

## Compilation Pipeline

The [`compile`](compile) script at the repository root converts the two checkpoints into a **TACHY-RT** deployable bundle. It orchestrates deploy, ONNX export, layer/block compilation, and runtime packaging using tools under `TACHY-Compiler/`.

### One-time setup (after cloning)

Run these steps **once** per clone—not on every compile:

```bash
# Duplicate model definitions for the compile-time graph
cp -rf ./models ./models_compile
cp -f ./models_compile/yolo_compile.py ./models_compile/yolo.py

# Symlink shared utilities into the YOLOv9 platform converter
cd ./TACHY-Compiler/platform_converter/utils/yolov9/
ln -sf ../../../../utils utils
ln -sf ../../../../models_compile models
cd -

# Allow execution of the native 4-bit block packer
chmod +x TACHY-Compiler/compiler/utils/block_4bit.out
```

### Configure checkpoint paths

Edit the user settings at the top of [`compile`](compile):

- **`PRE_PARAM_DIR`** — weights directory for the Stage 1 (pre-trained) model (`best.pt`)
- **`OPT_PARAM_DIR`** — weights directory for the Stage 2 (optimized) model (`best.pt`)

Both directories should contain a `best.pt` file produced by training.

### Run compilation

From the repository root:

```bash
python compile
```

The script creates a timestamped output folder (format `20YYMMDD_HHMMSS`) containing runtime artifacts, including a **`.tachyrt`** file. Verify that this file is present after a successful run.

---

## Quantization Model

Deeper-I tooling uses **XWN (eXtreme Weight Network)** quantization—not conventional INT4/INT8 post-training quantization. During training, custom convolution layers from `DDesignerAPI` carry bit-width and scale options; during compilation, weights are packed into sign, magnitude, scale, and header components for the target NPU.

---

## Known Fixes in This Version

An earlier compile configuration used a fixed default padding mode that caused **incorrect fixed bounding boxes** at inference. This repository sets block padding to dynamic mode:

```python
B_DEFAULT_PAD_MODE="--default_pad_mode=dynamic"
```

That change is already applied in [`compile`](compile) and resolves the bounding-box issue.

---

## Quick Checklist

- [ ] Conda env `umz_second` activated and requirements installed  
- [ ] Stage 1 training complete → `best.pt` saved  
- [ ] Stage 2 training complete → second `best.pt` saved  
- [ ] One-time compile symlinks and `chmod` applied  
- [ ] `PRE_PARAM_DIR` and `OPT_PARAM_DIR` updated in `compile`  
- [ ] `python compile` finished → `.tachyrt` present in output folder  

---

## Related Files

| File | Description |
|------|-------------|
| [`about_thi_folder.md`](about_thi_folder.md) | Informal project notes (source for this guide) |
| [`MODIFY.md`](MODIFY.md) | Changelog of model and training modifications |
| [`compile`](compile) | End-to-end deploy and TACHY-RT build script |
| [`requirements_umz_second.txt`](requirements_umz_second.txt) | Environment dependencies |
