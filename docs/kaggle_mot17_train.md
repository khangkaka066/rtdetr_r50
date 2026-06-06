# Kaggle MOT17 Fine-tuning

Use this when the Kaggle dataset is mounted at:

```text
/kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train
/kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test
```

Only the `train` split has `gt/gt.txt`, so it is the split used for fine-tuning. The `test` split can be used later for inference, but not supervised training.

## 1. Clone

```bash
git clone https://github.com/khangkaka066/rtdetr_r50.git
cd rtdetr_r50/rtdetr_pytorch
```

## 2. Install

Kaggle usually already has PyTorch. Install only the missing packages first:

```bash
pip install pycocotools scipy PyYAML onnx onnxruntime
```

If `torchvision.datapoints` is missing, install the repo-pinned versions:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
```

Restart the Kaggle notebook kernel after changing torch/torchvision.

## 3. Link Kaggle Input

The config expects images under `dataset/mot17/raw`, so create a symlink:

```bash
mkdir -p dataset/mot17
ln -s /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train dataset/mot17/raw
```

Check:

```bash
find dataset/mot17/raw -maxdepth 2 -type d | head
```

You should see folders like:

```text
dataset/mot17/raw/MOT17-02-FRCNN
dataset/mot17/raw/MOT17-04-FRCNN
```

## 4. Convert MOT17 to COCO

```bash
python tools/convert_mot17_to_coco.py \
  --mot-root dataset/mot17/raw \
  --out-dir dataset/mot17/annotations \
  --detector-variant FRCNN \
  --split half \
  --include-classes 1 \
  --min-visibility 0.1
```

This writes:

```text
dataset/mot17/annotations/train.json
dataset/mot17/annotations/val.json
```

## 5. Train on T4

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --amp
```

For Kaggle T4 memory, if you hit OOM, lower batch size in:

```text
configs/dataset/mot17_detection.yml
```

Change both train and val `batch_size` from `4` to `2`.

## 6. Save Outputs

Kaggle working files are saved under:

```text
output/rtdetr_r50vd_6x_mot17/
```

Important files:

```text
checkpoint.pth
log.txt
eval/
```
