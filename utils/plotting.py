"""Training curve plotting (extracted from train.py)."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils.evaluate import CLASS_NAMES
from utils.history import HistoryTracker


def plot_curves(history: HistoryTracker, save_dir: Path, epoch: int = None):
    curves_dir = save_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    train_epochs = list(range(len(history.records["train/loss"])))
    val_epochs = history.epochs
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Loss
    ax = axes[0, 0]
    ax.plot(train_epochs, history.records["train/loss"], "k-", alpha=0.5, lw=1.0, label="Train total")
    ax.plot(train_epochs, history.records["train/np_total"], "#1f77b4", alpha=0.6, lw=0.6, label="Train NP")
    ax.plot(train_epochs, history.records["train/hv"], "#ff7f0e", alpha=0.6, lw=0.6, label="Train HV")
    ax.plot(train_epochs, history.records["train/nc_total"], "#d62728", alpha=0.6, lw=0.6, label="Train NC")
    if val_epochs:
        ax.plot(val_epochs, history.records["val/loss"], "ko-", ms=6, label="Val total")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss — NP + HV + NC")
    ax.legend(fontsize=7, ncol=2); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Top-right: PQ + IoU
    def _plot_safe(ax, x_all, ys, label, *args, **kwargs):
        n = len(ys)
        if n > 0:
            ax.plot(x_all[:n], ys, *args, label=label, **kwargs)

    ax = axes[0, 1]
    if val_epochs:
        val_mPQ = history.records.get("val/mPQ_Tiss", history.records.get("val/mPQ", []))
        val_bPQ = history.records.get("val/bPQ_Tiss", history.records.get("val/bPQ", []))
        _plot_safe(ax, val_epochs, val_mPQ, "mPQ_Tiss", "r-o", ms=3, lw=1.2)
        _plot_safe(ax, val_epochs, val_bPQ, "bPQ_Tiss", "m-^", ms=3, lw=1.0, alpha=0.8)
        _plot_safe(ax, val_epochs, history.records.get("val/np_iou",[]), "NP IoU", "b-s", ms=2, lw=0.6, alpha=0.5)
        _plot_safe(ax, val_epochs, history.records.get("val/fg_miou",[]), "NC mIoU", "g-o", ms=2, lw=0.6, alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("PQ / IoU"); ax.set_ylim(0, 1)
    ax.set_title("mPQ_Tiss / bPQ_Tiss  |  NP IoU  |  NC mIoU")
    ax.legend(fontsize=7); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Bottom-left: Per-class IoU
    ax = axes[1, 0]
    if val_epochs and history.records["val/per_class_iou"]:
        per_class = np.array(history.records["val/per_class_iou"])
        n = len(per_class)
        for c in range(len(CLASS_NAMES)):
            ax.plot(val_epochs[:n], per_class[:, c], "o-", color=colors[c], ms=4, lw=1.0, label=CLASS_NAMES[c])
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU"); ax.set_ylim(0, 1)
    ax.set_title("Per-Class IoU")
    ax.legend(fontsize=7, ncol=3); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Bottom-right: LR
    ax = axes[1, 1]
    ax.plot(train_epochs, history.records["lr"], "k-", lw=1.0, label="LR")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_yscale("log")
    ax.set_title("Learning Rate")
    ax.legend(fontsize=8); ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    title = f"Training Curves (epoch {epoch})" if epoch is not None else "Training Curves"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(curves_dir / "curves.png", dpi=100)
    plt.close(fig)
    print(f"Curves → {curves_dir / 'curves.png'}")
