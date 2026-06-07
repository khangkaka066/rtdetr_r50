# Kaggle MOT17 Fine-tuning

Use this when the Kaggle dataset is mounted at:

```text
/kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train
/kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test
```

Only the `train` split has `gt/gt.txt`, so it is the split used for fine-tuning. The `test` split can be used later for inference, but not supervised training.

## 1. Clone

```bash
cd /kaggle/working

if [ ! -d rtdetr_r50 ]; then
  git clone https://github.com/khangkaka066/rtdetr_r50.git
else
  cd rtdetr_r50
  git pull
  cd ..
fi

cd rtdetr_r50/rtdetr_pytorch
```

## 2. Install

Kaggle already has PyTorch. Install the repo requirements; this repo no longer pins old torch versions:

```bash
pip install -r requirements.txt
```

## 3. One-command Path Setup

The repo uses a stable virtual dataset path:

```text
dataset/mot17/raw
```

You only need to provide the real dataset path once. The script creates the symlink and converts MOT17 annotations to COCO:

```bash
python tools/prepare_mot17_dataset.py \
  /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train \
  --force
```

If the dataset path changes later, run the same command with the new path:

```bash
python tools/prepare_mot17_dataset.py /new/path/to/MOT17/train --force
```

## 4. Train on T4

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --mot-root /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train \
  --amp \
  --epochs 24 \
  --eval-interval 3
```

`--mot-root` prepares the virtual path and COCO JSON automatically, so you do not need a separate prepare cell.
`--eval-interval 3` runs validation every 3 epochs and on the final epoch. Use `--eval-interval 1` for validation every epoch.

Or prepare and train in one command:

```bash
bash tools/train_mot17_from_path.sh \
  /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train \
  --epochs 24
```

Change `24` to any number you want. For a quick test, use `--epochs 1` or `--epochs 3`.

Training and evaluation print a text progress bar in the notebook, for example:

```text
Epoch: [0] [######------------------------] 120/665  18.0% elapsed=0:02:10 eta=0:09:48 loss=3.4210 lr=1.00e-05
```

During train, the bar displays current `loss` and `lr`; after each epoch, the code prints total epoch time.

The first command writes:

```text
dataset/mot17/annotations/train.json
dataset/mot17/annotations/val.json
```

For Kaggle T4 memory, if you hit OOM, lower batch size in:

```text
configs/dataset/mot17_detection.yml
```

Change both train and val `batch_size` from `4` to `2`.

## 5. Save Outputs

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

## 6. Run MOT Tracker

After training or after adding a saved Kaggle Model checkpoint, run the hybrid tracker on one MOT17 sequence:

```bash
python tools/track_mot.py \
  --source /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test/MOT17-08-FRCNN \
  -r /kaggle/input/models/nguyenvohoangkhang/rtdetr-r50-24epoch/pytorch/default/1/checkpoint.pth \
  --amp \
  --video-output output/MOT17-08-FRCNN.mp4 \
  --image-size 640 \
  --det-score 0.10 \
  --track-score 0.45 \
  --low-track-score 0.10 \
  --nms-iou 0.60 \
  --output output/MOT17-08-FRCNN.txt
```

To save annotated frames:

```bash
python tools/track_mot.py \
  --source /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test/MOT17-08-FRCNN \
  -r /kaggle/input/models/nguyenvohoangkhang/rtdetr-r50-24epoch/pytorch/default/1/checkpoint.pth \
  --amp \
  --vis-dir output/vis/MOT17-08-FRCNN \
  --det-score 0.10 \
  --track-score 0.45 \
  --nms-iou 0.60 \
  --output output/MOT17-08-FRCNN.txt
```

The tracker writes MOTChallenge rows:

```text
frame,id,x,y,w,h,score,-1,-1,-1
```

Create a download link in Kaggle:

```python
from IPython.display import FileLink
FileLink("/kaggle/working/rtdetr_r50/rtdetr_pytorch/output/MOT17-08-FRCNN.mp4")
```
