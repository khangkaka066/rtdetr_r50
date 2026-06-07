# RT-DETR R50VD Only

This checkout was pruned from `lyuwenyu/RT-DETR` to keep only the PyTorch code path for `rtdetr_r50vd`.

Kept:

- `rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml`
- `rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml`
- shared PyTorch training, evaluation, export, data, loss, optimizer, and RT-DETR source code
- `PResNet` backbone code required by `rtdetr_r50vd`

Removed:

- Paddle implementation
- RT-DETRv2 implementation
- benchmark utilities
- RegNet, DLA, R18, R34, R50-m, and R101 configs/backbone-only code paths

## Quick Start

```bash
cd rtdetr_pytorch
pip install -r requirements.txt
```

Prepare COCO:

```text
rtdetr_pytorch/dataset/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

Or edit the dataset paths in `configs/dataset/coco_detection.yml`.

Train:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

Evaluate:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint.pth --test-only
```

Track MOT17 with the RT-DETR checkpoint:

```bash
python tools/track_mot.py \
  --source /path/to/MOT17-08-FRCNN \
  -r /path/to/checkpoint.pth \
  --output output/MOT17-08-FRCNN.txt
```

See `docs/hybrid_mot_tracker.md` for the Kalman + xLSTM/LNN hybrid tracker pipeline and Kaggle commands.

Export ONNX:

```bash
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint.pth --check
```
