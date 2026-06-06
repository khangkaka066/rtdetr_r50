# Fine-tune RT-DETR R50VD on MOT17

The Kaggle dataset is MOT17-style data. RT-DETR expects COCO detection JSON, so first convert MOT17 `gt.txt` annotations to COCO.

## 1. Download

With Kaggle CLI:

```bash
cd /Users/nguyenvokhang/Downloads/LatenMOT/RT-DETR-r50vd/rtdetr_pytorch
mkdir -p dataset/mot17/raw
kaggle datasets download -d wenhoujinjust/mot-17 -p dataset/mot17/raw --unzip
```

If Kaggle extracts another nested folder, keep `dataset/mot17/raw` as the root that contains folders like `MOT17-02-FRCNN` or `train/MOT17-02-FRCNN`.

If you already added the Kaggle dataset as an input and it is mounted at `/kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train`, use the prepare script instead of downloading:

```bash
cd /kaggle/working/rtdetr_r50/rtdetr_pytorch
python tools/prepare_mot17_dataset.py \
  /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train \
  --force
```

See `kaggle_mot17_train.md` for the Kaggle-specific workflow.

## 2. Convert MOT17 to COCO

Default conversion:

```bash
python tools/convert_mot17_to_coco.py \
  --mot-root dataset/mot17/raw \
  --out-dir dataset/mot17/annotations \
  --detector-variant FRCNN \
  --split half \
  --include-classes 1 \
  --min-visibility 0.1
```

Notes:

- MOT17 has repeated sequence variants: `DPM`, `FRCNN`, and `SDP`. For detector fine-tuning, use one variant, usually `FRCNN`, to avoid duplicate frames.
- The converter maps MOT pedestrian class `1` to COCO `category_id=0`.
- This matches `num_classes: 1` and `remap_mscoco_category: False`.

## 3. Fine-tune

Fine-tune from the official COCO checkpoint:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --amp
```

Without AMP:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

## 4. Evaluate

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -r output/rtdetr_r50vd_6x_mot17/checkpoint.pth \
  --test-only
```

This trains a person detector, not a full tracker. Use its detections as input to your MOT tracker.
