# Tools

Train:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

Train with AMP:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --amp
```

Override epochs without editing YAML:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml --epochs 12
```

Run validation less often to speed up training:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml --epochs 12 --eval-interval 3
```

Resume:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint.pth
```

Fine-tune from official checkpoint:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

Evaluate:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint.pth --test-only
```

Export:

```bash
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint.pth --check
```
