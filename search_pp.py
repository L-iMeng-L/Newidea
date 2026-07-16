#!/usr/bin/env python3
"""
Post-processing hyperparameter search to maximise mPQ / bPQ.

Methods:  grid  |  bayes (Optuna TPE)

Searches over:
    np_thresh      — NP probability threshold
    min_area       — minimum instance area
    energy_thresh  — watershed energy threshold
    sobel_ksize    — Sobel kernel size (odd)
    marker_ksize   — morphological opening kernel

Usage:
    python search_pp.py /path/to/best.pth --method bayes --trials 200
    python search_pp.py /path/to/best.pth --method grid

Output:  search_results.json + sorted table
"""
import argparse
import json
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import create_model
from data import get_pannuke_loaders
from utils.postprocess import postprocess_nuclei
from utils.evaluate import evaluate_image_official, aggregate_metrics, load_tissue_types

CLASS_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]


def parse_args():
    p = argparse.ArgumentParser(description="Post-processing hyperparameter search")
    p.add_argument("checkpoint", type=str, help="Path to best.pth")
    p.add_argument("--output", type=str, default="./search_results.json")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_root", type=str, default="/home/lwy/dataset/PanNuke/processed")
    p.add_argument("--val_fold", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=18)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--encoder", type=str, default="uni2-h",
                   choices=["uni2-h", "tiny", "small", "base"],
                   help="Encoder variant (use 'tiny'/'base' for KD student models)")
    p.add_argument("--decoder_type", type=str, default="shared_unet",
                   choices=["shared_unet", "shared_unet_mala", "unet3", "unet3_mala"])
    p.add_argument("--match_iou", type=float, default=0.5)

    p.add_argument("--method", type=str, default="bayes", choices=["grid", "bayes"])
    # Grid
    p.add_argument("--np_thresh", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--min_area", type=str, default="5,10,20,40,80")
    p.add_argument("--energy_thresh", type=str, default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--sobel_ksize", type=str, default="11,21,31")
    p.add_argument("--marker_ksize", type=str, default="3,5,7")
    # Bayes
    p.add_argument("--trials", type=int, default=200,
                   help="Number of Optuna trials")
    p.add_argument("--seed", type=int, default=42,
                   help="Optuna sampler seed")
    return p.parse_args()


def parse_range(s, dtype=int):
    return [dtype(float(x)) if dtype is int else float(x) for x in s.split(",")]


@torch.no_grad()
def run_inference(model, loader, device, max_samples=0):
    model.eval()
    all_data = []
    count = 0
    for batch in tqdm(loader, desc="Inference", unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)

        np_probs = torch.sigmoid(outputs["np"]).cpu().numpy()
        hv_maps = outputs["hv"].cpu().numpy()
        nc_logits = outputs["nc"].cpu().numpy()

        for b in range(images.shape[0]):
            all_data.append({
                "np_prob": np_probs[b, 0],
                "hv_map": hv_maps[b],
                "nc_pixel": np.argmax(nc_logits[b], axis=0),   # precompute
                "mask": batch["mask"][b].cpu().numpy(),
                "inst_gt": batch["inst_gt"][b].cpu().numpy(),
            })
            count += 1
            if max_samples > 0 and count >= max_samples:
                return all_data
    return all_data


def _eval_one(args):
    """Single-image postprocess + PQ (module-level for pickling)."""
    d, params, match_iou = args
    inst_map, class_map, _ = postprocess_nuclei(
        d["np_prob"], d["hv_map"], d["nc_pixel"],
        np_thresh=params["np_thresh"],
        min_area=params["min_area"],
        energy_thresh=params["energy_thresh"],
        sobel_ksize=params["sobel_ksize"],
        marker_ksize=params["marker_ksize"],
    )
    true_type = d["mask"].copy()
    true_type[true_type >= 5] = 0
    pred_type = class_map.copy()
    pred_type[pred_type > 0] -= 1
    return evaluate_image_official(
        d["inst_gt"], true_type, inst_map, pred_type, match_iou=match_iou)


def evaluate_params(data, params, match_iou=0.5, num_workers=12, tissue_types=None):
    tasks = [(d, params, match_iou) for d in data]
    n = len(tasks)
    workers = min(num_workers, n)
    results = [None] * n
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, t): i for i, t in enumerate(tasks)}
        for f in tqdm(as_completed(futures), total=n, desc="  PP eval",
                       unit="img", leave=False):
            idx = futures[f]
            results[idx] = f.result()
    return aggregate_metrics(results, tissue_types=tissue_types)


def grid_search(data, args, tissue_types=None):
    np_thresh_vals = parse_range(args.np_thresh, dtype=float)
    min_area_vals = parse_range(args.min_area, dtype=int)
    energy_vals = parse_range(args.energy_thresh, dtype=float)
    sobel_vals = parse_range(args.sobel_ksize, dtype=int)
    marker_vals = parse_range(args.marker_ksize, dtype=int)

    total = (len(np_thresh_vals) * len(min_area_vals) *
             len(energy_vals) * len(sobel_vals) * len(marker_vals))
    print(f"Grid: {len(np_thresh_vals)}×{len(min_area_vals)}×{len(energy_vals)}×"
          f"{len(sobel_vals)}×{len(marker_vals)} = {total} combinations")

    combo_iter = itertools.product(
        np_thresh_vals, min_area_vals, energy_vals, sobel_vals, marker_vals)
    all_results = []
    for nt, ma, et, sk, mk in tqdm(combo_iter, total=total, unit="combo"):
        params = {"np_thresh": nt, "min_area": ma, "energy_thresh": et,
                  "sobel_ksize": sk, "marker_ksize": mk}
        metrics = evaluate_params(data, params, match_iou=args.match_iou,
                                  tissue_types=tissue_types)
        all_results.append({**params,
                            "bPQ": round(metrics["bPQ_Tiss"], 5),
                            "mPQ": round(metrics["mPQ_Tiss"], 5)})
    return all_results


def bayes_search(data, args, tissue_types=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        sk = trial.suggest_int("sobel_ksize", 5, 31, step=2)
        try:
            m = evaluate_params(data, {
                "np_thresh": trial.suggest_float("np_thresh", 0.1, 0.9, step=0.05),
                "min_area": trial.suggest_int("min_area", 2, 20),
                "energy_thresh": trial.suggest_float("energy_thresh", 0.05, 0.7, step=0.05),
                "sobel_ksize": sk,
                "marker_ksize": trial.suggest_int("marker_ksize", 1, 13, step=2),
            }, match_iou=args.match_iou, tissue_types=tissue_types)
        except Exception:
            return -1.0  # invalid params → worst score
        trial.set_user_attr("bPQ", m["bPQ_Tiss"])
        trial.set_user_attr("bPQ_img", m["PQb"])
        trial.set_user_attr("DQb", m["DQb"])
        trial.set_user_attr("SQb", m["SQb"])
        return m["mPQ_Tiss"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    pbar = tqdm(total=args.trials, desc="Bayes search", unit="trial")
    best_so_far = 0.0

    def callback(_study, trial):
        nonlocal best_so_far
        if trial.values and trial.values[0] > best_so_far:
            best_so_far = trial.values[0]
            pbar.set_postfix(best_mPQ=f"{best_so_far:.4f}")
        pbar.update(1)

    study.optimize(objective, n_trials=args.trials, callbacks=[callback],
                   show_progress_bar=False)
    pbar.close()

    all_results = []
    for t in study.trials:
        if t.values is None:
            continue
        all_results.append({
            **t.params,
            "mPQ": round(t.values[0], 5),
            "bPQ": round(t.user_attrs["bPQ"], 5),
        })

    return all_results


def report(all_results, output_path, checkpoint, data, args, tissue_types=None):
    from utils.evaluate import save_metrics

    all_results.sort(key=lambda x: x["mPQ"], reverse=True)
    best_mPQ = all_results[0]
    all_results.sort(key=lambda x: x["bPQ"], reverse=True)
    best_bPQ = all_results[0]

    # ---- Generate full metrics for best mPQ_Tiss config ----
    out_dir = str(Path(output_path).parent)
    best_params = {k: v for k, v in best_mPQ.items() if k not in ("bPQ", "mPQ")}
    metrics = evaluate_params(data, best_params, match_iou=args.match_iou,
                              tissue_types=tissue_types)
    save_metrics(metrics, out_dir, "fold3")
    print(f"Full metrics → {out_dir}/metrics_fold3.txt")

    print(f"\n{'='*70}")
    print(f"Best by bPQ_Tiss: {best_bPQ['bPQ']:.4f}  → { {k: v for k,v in best_bPQ.items() if k not in ('bPQ','mPQ')} }")
    print(f"Best by mPQ_Tiss: {best_mPQ['mPQ']:.4f}  → { {k: v for k,v in best_mPQ.items() if k not in ('bPQ','mPQ')} }")
    print(f"{'='*70}")

    all_results.sort(key=lambda x: x["mPQ"], reverse=True)
    print(f"\nTop-10 by mPQ_Tiss:")
    header = f"{'np_thresh':>10} {'min_area':>9} {'energy':>10} {'sobel':>7} {'marker':>7} {'mPQ_Tiss':>9} {'bPQ_Tiss':>9}"
    print(header); print("-" * 69)
    for r in all_results[:10]:
        print(f"{r['np_thresh']:>10.2f} {r['min_area']:>9} {r['energy_thresh']:>10.2f} "
              f"{r['sobel_ksize']:>7} {r['marker_ksize']:>7} {r['mPQ']:>9.4f} {r['bPQ']:>9.4f}")

    all_results.sort(key=lambda x: x["bPQ"], reverse=True)
    print(f"\nTop-10 by bPQ_Tiss:")
    header2 = f"{'np_thresh':>10} {'min_area':>9} {'energy':>10} {'sobel':>7} {'marker':>7} {'bPQ_Tiss':>9} {'mPQ_Tiss':>9}"
    print(header2); print("-" * 69)
    for r in all_results[:10]:
        print(f"{r['np_thresh']:>10.2f} {r['min_area']:>9} {r['energy_thresh']:>10.2f} "
              f"{r['sobel_ksize']:>7} {r['marker_ksize']:>7} {r['bPQ']:>9.4f} {r['mPQ']:>9.4f}")

    output = {
        "checkpoint": checkpoint,
        "best_bPQ_Tiss": best_bPQ["bPQ"],
        "best_params_bPQ": {k: v for k, v in best_bPQ.items() if k not in ("bPQ", "mPQ")},
        "best_mPQ_Tiss": best_mPQ["mPQ"],
        "best_params_mPQ": {k: v for k, v in best_mPQ.items() if k not in ("bPQ", "mPQ")},
        "all_results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {output_path}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Method: {args.method}")

    # ---- Load model ----
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_uni2 = (args.encoder == "uni2-h")
    model = create_model(
        variant=args.encoder, num_nc_classes=5,
        pretrained=(not is_uni2),  # ConvNeXt uses pretrained, UNI2-h loads from .bin
        decoder_type=args.decoder_type, enc_dropout=0.5, dec_dropout=0.2,
        freeze_encoder=is_uni2, full_unfreeze=is_uni2,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"epoch {ckpt.get('epoch', '?')}")

    # ---- Inference ----
    from data.pannuke import PanNukeDataset
    val_dataset = PanNukeDataset(data_root=args.data_root, folds=(args.val_fold,), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"\nRunning inference on fold-{args.val_fold} ...")
    t0 = time.time()
    data = run_inference(model, val_loader, device, max_samples=args.max_samples)
    print(f"  {len(data)} images, {time.time() - t0:.1f}s")

    # ---- Tissue types (official PanNuke protocol) ----
    tissue_types = load_tissue_types(args.data_root, args.val_fold)
    if tissue_types:
        print(f"  Tissue types loaded: {len(set(tissue_types))} tissues")

    # ---- Search ----
    if args.method == "grid":
        all_results = grid_search(data, args, tissue_types=tissue_types)
    else:
        all_results = bayes_search(data, args, tissue_types=tissue_types)

    report(all_results, args.output, args.checkpoint, data, args, tissue_types=tissue_types)


if __name__ == "__main__":
    main()
