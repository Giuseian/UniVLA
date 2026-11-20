"""
Experiment 1 - Stability of the latent action codes (our own metric, not in the paper).

The released stage-2 LAM gives two sets of codes for each pair of frames (4 tokens each, 16 possible values):
  - indices           : task-centric codes (the ones the policy predicts)
  - indices_uncontrol : task-irrelevant codes (frozen stage-1 codebook)

Claim we want to check: task-centric codes describe the robot's action and ignore task-irrelevant
visual changes. We take frame pairs (t, t+11) from the LIBERO RLDS data (same gap as fine-tuning),
apply visual perturbations that do NOT change the action, and measure how often the codes stay the same.

  both_*   : same perturbation on both frames (appearance change)
  frame2_* : perturbation only on the 2nd frame (task-irrelevant change BETWEEN the frames:
             this is what the task-irrelevant codes are supposed to absorb)
  controls : identity (should be 100%), static pair (t,t), next window (t+11,t+22),
             random pair (chance level)

Usage (inside the container):
  python my_experiments/exp1_lam/lam_stability.py --suites goal long --pairs_per_suite 200
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

SUITES = {
    "spatial": "libero_spatial_no_noops",
    "object": "libero_object_no_noops",
    "goal": "libero_goal_no_noops",
    "long": "libero_10_no_noops",
}
GAP = 11  # fine-tuning pairs frame 0 with frame window_size-1 = 11
RES = 224


# ----------------------------------------------------------------------------- data
def load_triplets(root, ds_name, n, seed):
    """Returns uint8 array (N, 3, H, W, C): frames t, t+GAP, t+2*GAP, one triplet per episode."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    infos = sorted(glob.glob(os.path.join(root, ds_name, "*", "dataset_info.json")))
    if not infos:
        raise FileNotFoundError(f"Dataset not found: {root}/{ds_name}/<version>/dataset_info.json")
    builder = tfds.builder_from_directory(os.path.dirname(infos[-1]))
    rng = random.Random(seed)
    triplets = []
    for ep in builder.as_dataset(split="train"):
        imgs = [s["observation"]["image"] for s in ep["steps"].as_numpy_iterator()]
        if len(imgs) < 2 * GAP + 1:
            continue
        t = rng.randrange(0, len(imgs) - 2 * GAP)
        triplets.append(np.stack([
            np.asarray(Image.fromarray(imgs[k]).resize((RES, RES), Image.LANCZOS))
            for k in (t, t + GAP, t + 2 * GAP)
        ]))
    rng.shuffle(triplets)
    return np.stack(triplets[:n])


def to_float(x):  # uint8 (N,H,W,C) -> float (N,C,H,W) in [0,1]
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0


# ----------------------------------------------------------------------------- perturbations
def brightness(x, f):
    return (x * f).clamp(0, 1)


def color(x):
    return (x * torch.tensor([1.15, 0.95, 0.80]).view(1, 3, 1, 1)).clamp(0, 1)


def blur(x):
    return TF.gaussian_blur(x, kernel_size=9, sigma=2.0)


def noise(x, gen):
    return (x + 0.03 * torch.randn(x.shape, generator=gen)).clamp(0, 1)


def camera_shift(x, px=16):
    """Crop px pixels from top and left and resize back: small camera translation + zoom (~7%)."""
    return F.interpolate(x[..., px:, px:], size=(RES, RES), mode="bilinear", align_corners=False)


def make_perturbations(seed):
    gen = torch.Generator().manual_seed(seed)
    return {
        # sanity check: must give 100%
        "identity": lambda a, b: (a, b),
        # same change on both frames
        "both_brighter": lambda a, b: (brightness(a, 1.3), brightness(b, 1.3)),
        "both_darker": lambda a, b: (brightness(a, 0.7), brightness(b, 0.7)),
        "both_color": lambda a, b: (color(a), color(b)),
        "both_blur": lambda a, b: (blur(a), blur(b)),
        "both_noise": lambda a, b: (noise(a, gen), noise(b, gen)),
        # task-irrelevant change between the two frames
        "frame2_brighter": lambda a, b: (a, brightness(b, 1.3)),
        "frame2_color": lambda a, b: (a, color(b)),
        "frame2_camera_shift": lambda a, b: (a, camera_shift(b)),
    }


# ----------------------------------------------------------------------------- model
def load_lam(ckpt, device):
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    lam = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128, num_latents=16, patch_size=14,
        enc_blocks=12, dec_blocks=12, num_heads=12,
    )
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    lam.load_state_dict({k.replace("lam.", ""): v for k, v in sd.items()}, strict=True)
    return lam.to(device).eval()


@torch.no_grad()
def encode(lam, f0, f1, bs, device):
    """Returns (task-centric codes, task-irrelevant codes), each LongTensor (N, 4)."""
    task, unc = [], []
    for i in range(0, len(f0), bs):
        video = torch.stack([f0[i:i + bs], f1[i:i + bs]], dim=1).to(device)  # (B, 2, C, H, W)
        out = lam.vq_encode(video)
        task.append(out["indices"].reshape(-1, 4).cpu())
        unc.append(out["indices_uncontrol"].reshape(-1, 4).cpu())
    return torch.cat(task), torch.cat(unc)


def agreement(a, b):
    same = (a == b)
    return {"token": same.float().mean().item() * 100, "all4": same.all(1).float().mean().item() * 100}


# ----------------------------------------------------------------------------- figure
def save_examples(f0, f1, perts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = f0[:1], f1[:1]
    rows = [(name, *fn(a, b)) for name, fn in perts.items()]
    fig, axes = plt.subplots(len(rows), 2, figsize=(4, 2 * len(rows)))
    for r, (name, x0, x1) in enumerate(rows):
        for c, x in enumerate((x0, x1)):
            axes[r, c].imshow(x[0].permute(1, 2, 0).numpy())
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    axes[0, 0].set_title("identity\nframe t", fontsize=8, loc="left")
    axes[0, 1].set_title("frame t+11", fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    p.add_argument("--pairs_per_suite", type=int, default=200)
    p.add_argument("--data_root", default="/data/datasets/libero_rlds")
    p.add_argument("--lam_ckpt", default="/data/models/univla-latent-action-model/lam-stage-2.ckpt")
    p.add_argument("--out_dir", default="/data/eval_logs/exp1_lam")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[exp1] Loading LAM from {args.lam_ckpt} on {device}")
    lam = load_lam(args.lam_ckpt, device)

    rows = {}  # condition -> suite -> {"task": {...}, "unc": {...}}
    codes = {}
    for suite in args.suites:
        print(f"[exp1] === {suite}: loading {args.pairs_per_suite} frame pairs ===")
        tr = load_triplets(args.data_root, SUITES[suite], args.pairs_per_suite, args.seed)
        f0, f1, f2 = (to_float(tr[:, k]) for k in range(3))
        print(f"[exp1] {suite}: {len(f0)} pairs")

        clean_t, clean_u = encode(lam, f0, f1, args.batch_size, device)
        codes[f"{suite}/clean_task"], codes[f"{suite}/clean_unc"] = clean_t.numpy(), clean_u.numpy()

        perts = make_perturbations(args.seed)
        if suite == args.suites[0]:
            save_examples(f0, f1, perts, os.path.join(args.out_dir, "examples.png"))

        conds = {name: fn(f0, f1) for name, fn in perts.items()}
        conds["ctrl_static_pair"] = (f0, f0)      # no motion at all
        conds["ctrl_next_window"] = (f1, f2)      # a different (real) action
        for name, (a, b) in conds.items():
            t, u = encode(lam, a, b, args.batch_size, device)
            rows.setdefault(name, {})[suite] = {"task": agreement(clean_t, t), "unc": agreement(clean_u, u)}
            codes[f"{suite}/{name}_task"], codes[f"{suite}/{name}_unc"] = t.numpy(), u.numpy()

        perm = torch.roll(torch.arange(len(f0)), 1)  # every pair compared with a different pair
        rows.setdefault("ctrl_random_pair", {})[suite] = {
            "task": agreement(clean_t, clean_t[perm]), "unc": agreement(clean_u, clean_u[perm])}

    # average over suites
    for name in rows:
        rows[name]["all"] = {
            k: {m: float(np.mean([rows[name][s][k][m] for s in args.suites])) for m in ("token", "all4")}
            for k in ("task", "unc")
        }

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "results": rows}, f, indent=2)
    np.savez(os.path.join(args.out_dir, "codes.npz"), **codes)

    lines = [
        "# Experiment 1 - LAM code stability",
        "",
        f"Suites: {', '.join(args.suites)} | pairs per suite: {args.pairs_per_suite} | gap: {GAP} frames | seed: {args.seed}",
        "",
        "Agreement with the codes of the clean pair (%). token = share of the 4 tokens unchanged;",
        "all4 = share of pairs with all 4 tokens unchanged. Chance level: see ctrl_random_pair.",
        "",
        "| Condition | Task-centric token | Task-centric all4 | Task-irrelevant token | Task-irrelevant all4 |",
        "|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        a = r["all"]
        lines.append(f"| {name} | {a['task']['token']:.1f} | {a['task']['all4']:.1f} "
                     f"| {a['unc']['token']:.1f} | {a['unc']['all4']:.1f} |")
    lines.append("")
    for suite in args.suites:
        lines += [f"## {suite}", "", "| Condition | Task-centric token | Task-irrelevant token |", "|---|---|---|"]
        for name, r in rows.items():
            lines.append(f"| {name} | {r[suite]['task']['token']:.1f} | {r[suite]['unc']['token']:.1f} |")
        lines.append("")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(table)
    print(table)
    print(f"[exp1] Saved results.md, results.json, codes.npz, examples.png in {args.out_dir}")


if __name__ == "__main__":
    main()
