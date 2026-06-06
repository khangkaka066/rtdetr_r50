# rtdetr_r50vd PyTorch

This directory contains the PyTorch implementation needed to train, evaluate, infer, and export `rtdetr_r50vd`.

## Model

| Model | Dataset | Input Size | AP val | AP50 val | Params | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| `rtdetr_r50vd` | COCO | 640 | 53.1 | 71.2 | 42M | 108 |

Checkpoint from the original model zoo:

```text
https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

## Install

```bash
pip install -r requirements.txt
```

Kaggle already ships PyTorch. The requirements file intentionally does not pin `torch` or `torchvision`, so it works with newer Kaggle Python environments where `torch==2.0.1` is unavailable.

## Data

Download COCO 2017 train and val images:

```text
dataset/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

If your COCO folder is elsewhere, edit `configs/dataset/coco_detection.yml`.

## Training

Single GPU:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

Multiple GPUs:

```bash
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

Fine-tune from the official checkpoint:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

## Evaluation

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r path/to/checkpoint.pth \
  --test-only
```

## Inference

```bash
python tools/infer.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r path/to/checkpoint.pth \
  -f path/to/image.jpg \
  -d cuda:0
```

## Export

```bash
python tools/export_onnx.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r path/to/checkpoint.pth \
  --check
```

## Custom Data

Use COCO-format annotations, set `remap_mscoco_category: False` in `configs/dataset/coco_detection.yml`, update `num_classes`, then fine-tune with `-t`.

## MOT17 Fine-tuning

MOT17 needs to be converted from MOT `gt.txt` annotations to COCO detection JSON before training.

```bash
python tools/prepare_mot17_dataset.py /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train --force

python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --amp
```

See `../docs/mot17_finetune.md` for the full workflow.
