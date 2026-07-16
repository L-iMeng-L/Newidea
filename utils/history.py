"""Training history tracker (extracted from train.py)."""
import json
from collections import defaultdict

import numpy as np


class HistoryTracker:
    def __init__(self):
        self.records = defaultdict(list)
        self.epochs = []

    def log_train(self, epoch, total, np_loss, hv_loss, nc_loss, gate_loss, lr, gw):
        self.records["train/loss"].append(total)
        self.records["train/np_total"].append(np_loss)
        self.records["train/hv"].append(hv_loss)
        self.records["train/nc_total"].append(nc_loss)
        self.records["lr"].append(lr)

    def log_val(self, epoch, avg_loss, np_iou, fg_miou, per_class_iou):
        self.records["val/loss"].append(avg_loss)
        self.records["val/np_iou"].append(np_iou)
        self.records["val/fg_miou"].append(fg_miou)
        self.records["val/per_class_iou"].append(per_class_iou)
        self.epochs.append(epoch)

    def save_json(self, path):
        clean = {}
        for k, v in self.records.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], (np.floating, np.integer)):
                    clean[k] = [float(x) for x in v]
                elif isinstance(v[0], list):
                    clean[k] = [[float(x) for x in row] for row in v]
                else:
                    clean[k] = v
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
