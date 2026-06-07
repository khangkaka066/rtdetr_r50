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
  --det-score 0.10 \
  --track-score 0.45 \
  --low-track-score 0.10 \
  --nms-iou 0.60 \
  --output output/MOT17-08-FRCNN.txt
```

Add visualization frames:

```python
!python tools/track_mot.py \
  --source /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/test/MOT17-08-FRCNN \
  -r /kaggle/input/models/nguyenvohoangkhang/rtdetr-r50-24epoch/pytorch/default/1/checkpoint.pth \
  --amp \
  --vis-dir output/vis/MOT17-08-FRCNN \
  --det-score 0.10 \
  --track-score 0.45 \
  --nms-iou 0.60 \
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

## Evaluation

`MOTA`, `IDF1`, `HOTA`, `FN`, `FP`, and `IDs` require ground truth. `MOT17/test` does not include ground truth, so local evaluation must use `MOT17/train` sequences.

```bash
cd /kaggle/working
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -r TrackEval/requirements.txt

cd /kaggle/working/rtdetr_r50/rtdetr_pytorch
mkdir -p output/mot17_eval

python tools/track_mot.py \
  --source /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train/MOT17-02-FRCNN \
  -r /kaggle/input/models/nguyenvohoangkhang/rtdetr-r50-24epoch/pytorch/default/1/checkpoint.pth \
  --amp \
  --det-score 0.10 \
  --track-score 0.45 \
  --low-track-score 0.10 \
  --nms-iou 0.60 \
  --output output/mot17_eval/MOT17-02-FRCNN.txt

python tools/eval_mot17_trackeval.py \
  --mot-root /kaggle/input/datasets/wenhoujinjust/mot-17/MOT17/train \
  --results-dir output/mot17_eval \
  --trackeval-root /kaggle/working/TrackEval \
  --seqs MOT17-02-FRCNN
```

## Main Files

- `tools/track_mot.py`: Kaggle entrypoint.
- `src/tracker/rtdetr_detector.py`: RT-DETR config/checkpoint inference wrapper.
- `src/tracker/hybrid_tracker.py`: tracking loop, gating, Hungarian matching.
- `src/tracker/motion.py`: Kalman, xLSTM-style branch, LNN branch, fusion gate.

## Notes

- `--image-size 800` can improve small-person recall but costs more VRAM.
- Use `--det-score 0.10` with `--track-score 0.45` to keep low-confidence detections for association without creating too many new tracks.
- `--nms-iou 0.60` removes duplicated RT-DETR queries before tracking.
- Missing predicted tracks are not drawn by default. Add `--show-missing-tracks` only when debugging occlusion.
- `--motion-checkpoint path.pth` can load trained xLSTM/LNN residual weights.
- `--disable-neural-motion` runs only Kalman + IoU matching.
