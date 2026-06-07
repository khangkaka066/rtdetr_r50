import argparse
import shutil
import sys
from pathlib import Path


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def list_sequences(mot_root, seqs):
    if seqs:
        return [seq.strip() for seq in seqs.split(",") if seq.strip()]
    return sorted([p.name for p in mot_root.iterdir() if (p / "gt" / "gt.txt").exists()])


def prepare_trackeval_layout(args):
    mot_root = Path(args.mot_root)
    results_dir = Path(args.results_dir)
    work_dir = Path(args.work_dir)
    seqs = list_sequences(mot_root, args.seqs)

    gt_base = work_dir / "gt" / "mot_challenge" / f"{args.benchmark}-{args.split}"
    tracker_base = (
        work_dir
        / "trackers"
        / "mot_challenge"
        / f"{args.benchmark}-{args.split}"
        / args.tracker_name
        / "data"
    )
    tracker_base.mkdir(parents=True, exist_ok=True)
    seqmap_file = work_dir / "seqmaps" / f"{args.benchmark}-{args.split}-{args.tracker_name}.txt"
    seqmap_file.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    for seq in seqs:
        seq_dir = mot_root / seq
        gt_txt = seq_dir / "gt" / "gt.txt"
        seqinfo = seq_dir / "seqinfo.ini"
        result_txt = results_dir / f"{seq}.txt"
        if not gt_txt.exists():
            missing.append(str(gt_txt))
            continue
        if not result_txt.exists():
            missing.append(str(result_txt))
            continue

        link_or_copy(gt_txt, gt_base / seq / "gt" / "gt.txt")
        if seqinfo.exists():
            link_or_copy(seqinfo, gt_base / seq / "seqinfo.ini")
        link_or_copy(result_txt, tracker_base / f"{seq}.txt")

    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    seqmap_file.write_text("name\n" + "\n".join(seqs) + "\n")
    return seqs, gt_base.parent, tracker_base.parent.parent.parent, seqmap_file


def run_trackeval(args, gt_folder, trackers_folder, seqmap_file):
    trackeval_root = Path(args.trackeval_root)
    if not (trackeval_root / "trackeval").exists():
        raise FileNotFoundError(
            f"TrackEval package not found: {trackeval_root / 'trackeval'}\n"
            "Clone it first: git clone https://github.com/JonathonLuiten/TrackEval.git"
        )

    sys.path.insert(0, str(trackeval_root))
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_CONFIG": False,
        "DISPLAY_LESS_PROGRESS": True,
    })

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "BENCHMARK": args.benchmark,
        "SPLIT_TO_EVAL": args.split,
        "TRACKERS_TO_EVAL": [args.tracker_name],
        "SEQMAP_FILE": str(seqmap_file),
        "CLASSES_TO_EVAL": ["pedestrian"],
        "PRINT_CONFIG": False,
    })

    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    print("Running TrackEval with Python API")
    evaluator.evaluate(dataset_list, metrics_list)


def main():
    parser = argparse.ArgumentParser(description="Prepare and run TrackEval for MOT17 tracker results.")
    parser.add_argument("--mot-root", required=True, help="MOT17/train directory with sequence gt folders")
    parser.add_argument("--results-dir", required=True, help="directory containing MOT result txt files named SEQ.txt")
    parser.add_argument("--trackeval-root", default="/kaggle/working/TrackEval")
    parser.add_argument("--work-dir", default="output/trackeval_data")
    parser.add_argument("--tracker-name", default="rtdetr_r50_hybrid")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seqs", default="", help="comma-separated sequence names; default uses all with gt")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    seqs, gt_folder, trackers_folder, seqmap_file = prepare_trackeval_layout(args)
    print("Prepared sequences:")
    for seq in seqs:
        print(f"  - {seq}")
    print(f"GT_FOLDER={gt_folder}")
    print(f"TRACKERS_FOLDER={trackers_folder}")
    print(f"SEQMAP_FILE={seqmap_file}")

    if not args.prepare_only:
        run_trackeval(args, gt_folder, trackers_folder, seqmap_file)


if __name__ == "__main__":
    main()
