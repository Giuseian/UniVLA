"""
Experiment 4 - creates a subset of a LIBERO RLDS dataset (default: 10% of LIBERO-Goal).

The authors' code already lists `libero_goal_no_noops_mini` (configs.py / mixtures.py) but the
data was never released, so we build it: we keep the same fraction of episodes FOR EACH TASK
(so every task is still present), chosen at random with a fixed seed, and write them as a new
RLDS/TFDS dataset that the fine-tuning script can read by name.

Usage (inside the container / Apptainer image):
  python my_experiments/exp4_data_efficiency/make_libero_subset.py \
      --src libero_goal_no_noops --dst libero_goal_no_noops_mini --fraction 0.1
"""
import argparse
import glob
import json
import os
import random
import shutil
from collections import defaultdict

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


def instruction_of(raw):
    ex = tf.train.Example.FromString(raw.numpy())
    keys = [k for k in ex.features.feature if k.endswith("language_instruction")]
    assert keys, f"no language_instruction feature, keys: {list(ex.features.feature)[:20]}"
    return ex.features.feature[keys[0]].bytes_list.value[0].decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/data/datasets/libero_rlds")
    p.add_argument("--src", default="libero_goal_no_noops")
    p.add_argument("--dst", default="libero_goal_no_noops_mini")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    infos = sorted(glob.glob(os.path.join(args.data_root, args.src, "*", "dataset_info.json")))
    assert infos, f"source dataset not found in {args.data_root}/{args.src}"
    src_dir = os.path.dirname(infos[-1])
    version = os.path.basename(src_dir)
    dst_dir = os.path.join(args.data_root, args.dst, version)
    assert not os.path.exists(dst_dir), f"{dst_dir} already exists: delete it first"

    shards = sorted(glob.glob(os.path.join(src_dir, f"{args.src}-train.tfrecord-*")))
    assert shards, f"no train shards in {src_dir}"
    print(f"[subset] source: {src_dir} ({len(shards)} shards)")

    # 1. read all episodes (raw records) and group them by task
    by_task = defaultdict(list)
    for raw in tf.data.TFRecordDataset(shards):
        by_task[instruction_of(raw)].append(raw.numpy())
    total = sum(len(v) for v in by_task.values())
    print(f"[subset] {total} episodes, {len(by_task)} tasks")

    # 2. keep the same fraction of episodes for each task
    rng = random.Random(args.seed)
    kept, summary = [], {}
    for task in sorted(by_task):
        eps = by_task[task]
        idx = sorted(rng.sample(range(len(eps)), max(1, round(len(eps) * args.fraction))))
        kept += [eps[i] for i in idx]
        summary[task] = {"total": len(eps), "kept": len(idx), "kept_indices": idx}
        print(f"[subset]   {len(idx):3d} / {len(eps):3d}  {task}")
    rng.shuffle(kept)

    # 3. write them as a new TFDS dataset (1 shard is enough for a few hundred MB)
    os.makedirs(dst_dir)
    out_name = f"{args.dst}-train.tfrecord-00000-of-00001"
    with tf.io.TFRecordWriter(os.path.join(dst_dir, out_name)) as w:
        for raw in kept:
            w.write(raw)
    n_bytes = os.path.getsize(os.path.join(dst_dir, out_name))

    with open(os.path.join(src_dir, "dataset_info.json")) as f:
        info = json.load(f)
    info["name"] = args.dst
    for split in info["splits"]:
        if split["name"] == "train":
            split["shardLengths"] = [str(len(kept))]
            split["numBytes"] = str(n_bytes)
            split.pop("statistics", None)
    info["splits"] = [s for s in info["splits"] if s["name"] == "train"]
    with open(os.path.join(dst_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    for extra in glob.glob(os.path.join(src_dir, "features.json")):
        shutil.copy(extra, dst_dir)

    with open(os.path.join(dst_dir, "subset_info.json"), "w") as f:
        json.dump({"source": args.src, "fraction": args.fraction, "seed": args.seed,
                   "episodes_total": total, "episodes_kept": len(kept), "tasks": summary}, f, indent=2)

    # 4. check: the new dataset must be readable by name, like the fine-tuning script does
    import tensorflow_datasets as tfds
    builder = tfds.builder(args.dst, data_dir=args.data_root)
    n = sum(1 for _ in builder.as_dataset(split="train"))
    print(f"[subset] check: tfds.builder('{args.dst}') reads {n} episodes "
          f"({n / total * 100:.1f}% of {total}) -> {'OK' if n == len(kept) else 'MISMATCH'}")
    print(f"[subset] written to {dst_dir}")


if __name__ == "__main__":
    main()
