# Hybrid MOT Tracker

This repo now includes a runnable tracking pipeline:

```text
frame -> RT-DETR detections -> Kalman prediction
      -> xLSTM residual branch
      -> LNN residual branch
      -> fusion gate
      -> uncertainty-aware Hungarian matching
      -> active tracks in MOT format
```

The RT-DETR detector is fully loaded from the normal config and checkpoint.
The xLSTM/LNN residual branches are implemented as stateful PyTorch modules
with zero-initialized residual heads. That means the tracker is safe to run
without a separate motion checkpoint: it behaves like a Kalman/IoU tracker
until you train or load residual motion weights.

## Kaggle Run

```python
%cd /kaggle/working

!git clone https://github.com/khangkaka066/rtdetr_r50.git
%cd /kaggle/working/rtdetr_r50/rtdetr_pytorch

!pip install -r requirements.txt
```

Run one MOT17 sequence:

```python
!python tools/track_mot.py \
  --source /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test/MOT17-08-FRCNN \
  -r /kaggle/input/models/nguyenvohoangkhang/rtdetr-r50-24epoch/pytorch/default/1/checkpoint.pth \
  --amp \
  --video-output output/MOT17-08-FRCNN.mp4 \
  --image-size 640 \
  --det-score 0.35 \
  --output output/MOT17-08-FRCNN.txt
```

Add visualization frames:

```python
!python tools/track_mot.py \
  --source /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test/MOT17-08-FRCNN \
  -r /kaggle/input/models/nguyenvohoangkhang/rtdetr-r50-24epoch/pytorch/default/1/checkpoint.pth \
  --amp \
  --vis-dir output/vis/MOT17-08-FRCNN \
  --output output/MOT17-08-FRCNN.txt
```

The output follows MOTChallenge format:

```text
frame,id,x,y,w,h,score,-1,-1,-1
```

Download the MP4 from Kaggle:

```python
from IPython.display import FileLink
FileLink("/kaggle/working/rtdetr_r50/rtdetr_pytorch/output/MOT17-08-FRCNN.mp4")
```

## Main Files

- `tools/track_mot.py`: Kaggle entrypoint.
- `src/tracker/rtdetr_detector.py`: RT-DETR config/checkpoint inference wrapper.
- `src/tracker/hybrid_tracker.py`: tracking loop, gating, Hungarian matching.
- `src/tracker/motion.py`: Kalman, xLSTM-style branch, LNN branch, fusion gate.

## Notes

- `--image-size 800` can improve small-person recall but costs more VRAM.
- `--motion-checkpoint path.pth` can load trained xLSTM/LNN residual weights.
- `--disable-neural-motion` runs only Kalman + IoU matching.
