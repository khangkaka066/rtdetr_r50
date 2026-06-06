"""by lyuwenyu
"""

import os 
import sys 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
import argparse
import subprocess
import time
from pathlib import Path

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def prepare_mot17_if_needed(args):
    if not args.mot_root:
        return

    rank = int(os.environ.get('RANK', '0'))
    train_ann = Path('dataset/mot17/annotations/train.json')
    val_ann = Path('dataset/mot17/annotations/val.json')

    if rank != 0:
        print(f'Rank {rank}: waiting for MOT17 annotations prepared by rank 0...')
        for _ in range(600):
            if train_ann.exists() and val_ann.exists():
                print(f'Rank {rank}: MOT17 annotations are ready.')
                return
            time.sleep(1)
        raise TimeoutError('Timed out waiting for MOT17 annotations from rank 0.')

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepare_mot17_dataset.py'),
        args.mot_root,
        '--force',
    ]
    print('Preparing MOT17 dataset:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main(args, ) -> None:
    '''main
    '''
    prepare_mot17_if_needed(args)

    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    overrides = {}
    if args.epochs is not None:
        overrides['epoches'] = args.epochs
    if args.eval_interval is not None:
        overrides['eval_interval'] = args.eval_interval

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning,
        **overrides
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--epochs', '--epoches', dest='epochs', type=int,
                        help='override total training epochs from the config')
    parser.add_argument('--mot-root', type=str,
                        help='real MOT17 train directory; prepares dataset/mot17/raw and COCO JSON before training')
    parser.add_argument('--eval-interval', type=int,
                        help='run validation every N epochs; 0 disables validation except final epoch')
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)
