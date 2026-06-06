#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/train_mot17_from_path.sh /path/to/MOT17/train"
  exit 1
fi

DATASET_ROOT="$1"
shift || true

python tools/prepare_mot17_dataset.py "$DATASET_ROOT" --force

python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --amp \
  "$@"
