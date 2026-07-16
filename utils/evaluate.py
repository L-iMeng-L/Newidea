"""
Evaluation metrics for nuclei instance segmentation + classification.

Follows the OFFICIAL PanNuke evaluation protocol (Graham et al., 2019):
    1. Split instance map by class → per-class instance maps
    2. Compute get_fast_pq() on each class independently
    3. mPQ = nanmean of per-class PQ values

Binary PQ: binarize all 5 classes → single instance map → get_fast_pq.
"""
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


CLASS_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]
NUM_CLASSES = len(CLASS_NAMES)


# ==============================================================================
#  get_fast_pq  (official PanNuke implementation)
# ==============================================================================

def get_fast_pq(true, pred, match_iou=0.5):
    """
    DQ = TP / (TP + 0.5*FP + 0.5*FN)
    SQ = sum of paired IoU / TP
    PQ = DQ * SQ

    Args:
        true: [H, W] int, contiguous 0..N instance IDs
        pred: [H, W] int, contiguous 0..M instance IDs

    Returns:
        [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
    """
    assert match_iou >= 0.0, "Can't be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))

    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    # Pairwise IoU matrix
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = paired_iou.sum() / (tp + 1.0e-6) if tp > 0 else 0.0
    pq = dq * sq

    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def remap_label(pred, by_size=False):
    """Remap instance IDs to contiguous 1..N."""
    pred = np.asarray(pred, dtype=np.int32)
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_size = [(pred == i).sum() for i in pred_id]
        pred_id = [i for _, i in sorted(zip(pred_size, pred_id), reverse=True)]
    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


# ==============================================================================
#  Per-class instance PQ  (official PanNuke protocol)
# ==============================================================================

def split_inst_by_class(inst_map: np.ndarray, class_map: np.ndarray,
                        num_classes: int = 5) -> List[np.ndarray]:
    """
    Split a single instance map into per-class instance maps.

    Each output channel c contains ONLY instances whose predicted/GT class is c.
    """
    out = []
    for c in range(num_classes):
        ch = np.zeros_like(inst_map, dtype=np.int32)
        # Keep only instances of class c
        for iid in np.unique(inst_map):
            if iid == 0:
                continue
            mask = inst_map == iid
            if class_map[mask][0] == c:
                ch[mask] = iid
        out.append(remap_label(ch))
    return out


def evaluate_image_official(
    true_inst: np.ndarray,
    true_type: np.ndarray,
    pred_inst: np.ndarray,
    pred_type: np.ndarray,
    match_iou: float = 0.5,
) -> dict:
    """
    Evaluate a single image following the official PanNuke protocol.

    Three INDEPENDENT metrics:
        1. Binary PQ    — detection-only (all nuclei merged into one blob)
        2. Per-class PQ  — official protocol: split by class → per-class get_fast_pq
        3. Confusion     — instance-level matching on full instance maps

    Args:
        true_inst: [H, W] GT instance IDs (0=bg)
        true_type: [H, W] GT class per pixel (0..C-1, bg=0)
        pred_inst: [H, W] predicted instance IDs (0=bg)
        pred_type: [H, W] predicted class per pixel (0..C-1, bg=0)

    Returns:
        dict with per-class PQ values, binary PQ, confusion matrix, instance counts
    """
    true_inst = remap_label(true_inst)
    pred_inst = remap_label(pred_inst)

    C = NUM_CLASSES
    gt_n = true_inst.max()    # number of GT instances
    pred_n = pred_inst.max()  # number of predicted instances

    # ---- 1. Binary PQ (detection only, class-agnostic) ----
    # Use instance maps directly (already remapped above) — each nucleus keeps
    # its individual ID, same as official binarize() on per-class channels.
    if true_inst.max() == 0:
        [dq_b, sq_b, pq_b], _ = [np.nan, np.nan, np.nan], [[], [], [], []]
    else:
        [dq_b, sq_b, pq_b], _ = get_fast_pq(true_inst, pred_inst, match_iou)

    # ---- Per-class PQ: official protocol ----
    # Split instance map by class
    true_channels = split_inst_by_class(true_inst, true_type, C)
    pred_channels = split_inst_by_class(pred_inst, pred_type, C)

    per_class_pq = {}
    per_class_prf = {}
    for c in range(C):
        tc = true_channels[c]
        pc = pred_channels[c]
        if tc.max() == 0:
            per_class_pq[c] = {'DQ': np.nan, 'SQ': np.nan, 'PQ': np.nan}
            per_class_prf[c] = {'TP': 0, 'FP': 0, 'FN': 0,
                                'P': np.nan, 'R': np.nan, 'F1': np.nan}
        else:
            [dq, sq, pq], pair_info = get_fast_pq(tc, pc, match_iou)
            per_class_pq[c] = {'DQ': dq, 'SQ': sq, 'PQ': pq}
            paired_t, paired_p, unpaired_t, unpaired_p = pair_info
            tp = len(paired_t)
            fp = len(unpaired_p)
            fn = len(unpaired_t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1   = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else np.nan
            per_class_prf[c] = {'TP': tp, 'FP': fp, 'FN': fn,
                                'P': prec, 'R': rec, 'F1': f1}

    # ---- 3. Confusion matrix (independent instance-level matching) ----
    # Get per-instance GT class labels
    true_cls = {}
    for iid in np.unique(true_inst):
        if iid == 0: continue
        labels = true_type[true_inst == iid]
        labels = labels[(labels >= 0) & (labels < C)]
        if len(labels) > 0:
            true_cls[int(iid)] = int(np.bincount(labels.astype(np.int64)).argmax())

    # Get per-instance pred class labels
    pred_cls = {}
    for iid in np.unique(pred_inst):
        if iid == 0: continue
        labels = pred_type[pred_inst == iid]
        labels = labels[(labels >= 0) & (labels < C)]
        if len(labels) > 0:
            pred_cls[int(iid)] = int(np.bincount(labels.astype(np.int64)).argmax())

    # Match instances and build confusion
    # Use instance-level (not binary) matching so each nucleus is paired individually.
    confusion = np.zeros((C, C), dtype=np.int64)

    if true_inst.max() > 0 and pred_inst.max() > 0:
        _, pair_info = get_fast_pq(true_inst, pred_inst, match_iou)
        paired_true_ids, paired_pred_ids, _, _ = pair_info

        for t_id, p_id in zip(paired_true_ids, paired_pred_ids):
            tc = true_cls.get(t_id, None)
            pc = pred_cls.get(p_id, None)
            if tc is not None and pc is not None:
                confusion[tc, pc] += 1

    # ---- Diagnostic: per-class instance counts ----
    gt_per_class = np.zeros(C, dtype=np.int32)
    pred_per_class = np.zeros(C, dtype=np.int32)
    for c in range(C):
        gt_per_class[c] = true_channels[c].max()
        pred_per_class[c] = pred_channels[c].max()

    return {
        'dq_b': dq_b, 'sq_b': sq_b, 'pq_b': pq_b,
        'per_class_pq': per_class_pq,
        'per_class_prf': per_class_prf,
        'confusion': confusion,
        'gt_n': gt_n, 'pred_n': pred_n,
        'gt_per_class': gt_per_class, 'pred_per_class': pred_per_class,
    }


# ==============================================================================
#  Aggregation
# ==============================================================================

def aggregate_metrics(results: List[dict], tissue_types: List[str] = None) -> dict:
    """Aggregate per-image results following official PanNuke protocol.

    Per-class PQ: nanmean of per-image per-class PQ values
    Tissue-level: per-tissue bPQ/mPQ → nanmean across 19 tissues
    (same as official code: np.nanmean([pq for pq in mPQ_all]))
    """
    C = NUM_CLASSES

    dqb_vals = [r['dq_b'] for r in results if not np.isnan(r['dq_b'])]
    sqb_vals = [r['sq_b'] for r in results if not np.isnan(r['sq_b'])]
    pqb_vals = [r['pq_b'] for r in results if not np.isnan(r['pq_b'])]

    total_confusion = np.zeros((C, C), dtype=np.int64)
    class_pq_all = {c: [] for c in range(C)}
    class_dq_all = {c: [] for c in range(C)}
    class_sq_all = {c: [] for c in range(C)}

    total_gt = 0
    total_pred = 0
    total_gt_per_class = np.zeros(C, dtype=np.int64)
    total_pred_per_class = np.zeros(C, dtype=np.int64)

    class_prf_all = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(C)}

    for r in results:
        total_confusion += r['confusion']
        total_gt += r.get('gt_n', 0)
        total_pred += r.get('pred_n', 0)
        total_gt_per_class += r.get('gt_per_class', np.zeros(C, dtype=np.int32))
        total_pred_per_class += r.get('pred_per_class', np.zeros(C, dtype=np.int32))
        for c in range(C):
            pc = r['per_class_pq'][c]
            if not np.isnan(pc['PQ']):
                class_pq_all[c].append(pc['PQ'])
                class_dq_all[c].append(pc['DQ'])
                class_sq_all[c].append(pc['SQ'])
            prf = r.get('per_class_prf', {}).get(c, {})
            if prf and not np.isnan(prf.get('F1', np.nan)):
                class_prf_all[c]['TP'] += prf.get('TP', 0)
                class_prf_all[c]['FP'] += prf.get('FP', 0)
                class_prf_all[c]['FN'] += prf.get('FN', 0)

    def _nanmean(vals):
        return float(np.nanmean(vals)) if vals else np.nan

    metrics = {}
    metrics['DQb'] = _nanmean(dqb_vals)
    metrics['SQb'] = _nanmean(sqb_vals)
    metrics['PQb'] = _nanmean(pqb_vals)
    metrics['Fb'] = (2 * metrics['DQb'] * metrics['SQb'] /
                     (metrics['DQb'] + metrics['SQb'] + 1e-6)) if not np.isnan(metrics['DQb']) else np.nan

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            'DQ': _nanmean(class_dq_all[c]),
            'SQ': _nanmean(class_sq_all[c]),
            'PQ': _nanmean(class_pq_all[c]),
        }

    metrics['per_class'] = per_class
    metrics['DQm'] = _nanmean([v['DQ'] for v in per_class.values()])
    metrics['SQm'] = _nanmean([v['SQ'] for v in per_class.values()])
    pq_list = [_nanmean(class_pq_all[c]) for c in range(C)]
    metrics['mPQ'] = _nanmean(pq_list)
    metrics['bPQ'] = metrics['PQb']

    # Per-class P/R/F1 from accumulated TP/FP/FN
    per_class_prf = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = class_prf_all[c]['TP']
        fp = class_prf_all[c]['FP']
        fn = class_prf_all[c]['FN']
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1   = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else np.nan
        per_class_prf[name] = {'P': float(prec), 'R': float(rec), 'F1': float(f1)}
    metrics['per_class_prf'] = per_class_prf
    metrics['total_gt_instances'] = int(total_gt)
    metrics['total_pred_instances'] = int(total_pred)
    metrics['gt_per_class'] = total_gt_per_class.tolist()
    metrics['pred_per_class'] = total_pred_per_class.tolist()
    metrics['confusion_matrix'] = total_confusion.tolist()

    # ---- Tissue-level aggregation (official PanNuke protocol) ----
    # Each tissue type gets equal weight → nanmean across 19 tissues.
    # Falls back to per-image mean when tissue_types is not provided.
    if tissue_types is not None and len(tissue_types) == len(results):
        tissue_bPQ = {}
        tissue_mPQ = {}
        for r, tiss in zip(results, tissue_types):
            if not np.isnan(r.get('pq_b', np.nan)):
                tissue_bPQ.setdefault(tiss, []).append(r['pq_b'])
            img_pqs = []
            for c in range(C):
                pc = r['per_class_pq'][c]
                if not np.isnan(pc['PQ']):
                    img_pqs.append(pc['PQ'])
            if img_pqs:
                tissue_mPQ.setdefault(tiss, []).append(np.nanmean(img_pqs))

        per_tissue_bPQ = {t: _nanmean(v) for t, v in tissue_bPQ.items()}
        per_tissue_mPQ = {t: _nanmean(v) for t, v in tissue_mPQ.items()}

        metrics['per_tissue'] = {
            t: {'bPQ': per_tissue_bPQ.get(t, np.nan),
                'mPQ': per_tissue_mPQ.get(t, np.nan),
                'n_images': len(tissue_bPQ.get(t, []))}
            for t in sorted(set(list(tissue_bPQ.keys()) + list(tissue_mPQ.keys())))
        }
        metrics['bPQ_Tiss'] = _nanmean(list(per_tissue_bPQ.values()))
        metrics['mPQ_Tiss'] = _nanmean(list(per_tissue_mPQ.values()))
    else:
        # Fallback: per-image equal weight
        per_image_bPQ = [r['pq_b'] for r in results if not np.isnan(r.get('pq_b', np.nan))]
        metrics['bPQ_Tiss'] = _nanmean(per_image_bPQ)
        per_image_mPQ = []
        for r in results:
            img_pqs = []
            for c in range(C):
                pc = r['per_class_pq'][c]
                if not np.isnan(pc['PQ']):
                    img_pqs.append(pc['PQ'])
            if img_pqs:
                per_image_mPQ.append(float(np.nanmean(img_pqs)))
        metrics['mPQ_Tiss'] = _nanmean(per_image_mPQ)

    return metrics


# ==============================================================================
#  Save helpers
# ==============================================================================

def save_metrics(metrics: dict, save_dir: str, fold_name: str = ''):
    os.makedirs(save_dir, exist_ok=True)
    tag = f'_{fold_name}' if fold_name else ''

    json_path = os.path.join(save_dir, f'metrics{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    txt_path = os.path.join(save_dir, f'metrics{tag}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Results{(" on " + fold_name) if fold_name else ""}\n')
        f.write('=' * 80 + '\n')
        for k in ['DQb', 'SQb', 'bPQ', 'bPQ_Tiss', 'Fb', 'DQm', 'SQm', 'mPQ', 'mPQ_Tiss']:
            f.write(f'  {k:<10}: {metrics.get(k, "N/A")}\n')

        f.write('\nInstance counts:\n')
        f.write(f'  GT total:   {metrics.get("total_gt_instances", "N/A")}\n')
        f.write(f'  Pred total: {metrics.get("total_pred_instances", "N/A")}\n')
        f.write('  Per-class (GT → Pred):\n')
        for i, name in enumerate(CLASS_NAMES):
            gt_c = metrics.get('gt_per_class', [0]*5)[i]
            pd_c = metrics.get('pred_per_class', [0]*5)[i]
            f.write(f'    {name:<14}: GT={gt_c:5d}  Pred={pd_c:5d}\n')

        f.write('\nPer-class PQ:\n')
        for name, v in metrics.get('per_class', {}).items():
            f.write(f"  {name:<14}: DQ={v['DQ']:.4f}  SQ={v['SQ']:.4f}  PQ={v['PQ']:.4f}\n")

        f.write('\nPer-class P/R/F1:\n')
        f.write(f"{'Class':<14} {'P':>8} {'R':>8} {'F1':>8}\n")
        for name, v in metrics.get('per_class_prf', {}).items():
            f.write(f"  {name:<14}: {v['P']:>8.4f} {v['R']:>8.4f} {v['F1']:>8.4f}\n")

        per_tissue = metrics.get('per_tissue', {})
        if per_tissue:
            f.write('\nPer-tissue (official 19-organ equal-weight):\n')
            f.write(f'{"Tissue":<20} {"Imgs":>5} {"bPQ":>8} {"mPQ":>8}\n')
            f.write('-' * 44 + '\n')
            for t, v in per_tissue.items():
                f.write(f'{t:<20} {v["n_images"]:5d} {v["bPQ"]:8.4f} {v["mPQ"]:8.4f}\n')

        cm = metrics.get('confusion_matrix', None)
        if cm is not None:
            f.write('\nInstance Confusion Matrix (true\\pred):\n')
            f.write(' ' * 16 + ' '.join(f'{n[:10]:>10}' for n in CLASS_NAMES) + '\n')
            for i, row in enumerate(cm):
                f.write(f'{CLASS_NAMES[i]:>14}  ' + ' '.join(f'{x:10d}' for x in row) + '\n')

    print(f'Metrics saved → {json_path}')


# ==============================================================================
#  Tissue type helpers (official 19-tissue PanNuke protocol)
# ==============================================================================

def load_tissue_types(data_root: str, fold: int) -> list:
    """Load tissue type labels for all images in a validation fold.

    The PanNuke dataset records one of 19 tissue types per image.
    types.npy[i] maps to the i-th image (same order as processed PNG files).

    Returns:
        List[str] of length N, one tissue name per image.
    """
    import os
    types_path = os.path.join(data_root, f"Fold{fold}", "images", f"Fold{fold}", "types.npy")
    if not os.path.exists(types_path):
        # Try alternate path (original PanNuke structure)
        types_path = os.path.join(data_root.replace("/processed", ""), f"Fold{fold}",
                                  "images", f"Fold{fold}", "types.npy")
    if not os.path.exists(types_path):
        print(f"  Warning: types.npy not found at {types_path}, skipping tissue-level aggregation")
        return None
    return np.load(types_path).tolist()


# ==============================================================================
#  Parallel PQ evaluation (multiprocessing)
# ==============================================================================

def _eval_one_pq(args):
    """Single-image postprocess + evaluate (module-level for ProcessPoolExecutor)."""
    np_prob, hv, nc_logits, mask, inst_gt, np_thresh, min_area, \
        energy_thresh, sobel_ksize, marker_ksize, match_iou = args
    from utils.postprocess import postprocess_nuclei

    inst_map, class_map, _ = postprocess_nuclei(
        np_prob, hv, nc_logits,
        np_thresh=np_thresh, min_area=min_area,
        energy_thresh=energy_thresh, sobel_ksize=sobel_ksize,
        marker_ksize=marker_ksize,
    )
    true_type = mask.copy()
    true_type[true_type >= 5] = 0
    pred_type = class_map.copy()
    pred_type[pred_type > 0] -= 1
    return evaluate_image_official(
        inst_gt, true_type, inst_map, pred_type, match_iou=match_iou)


def evaluate_parallel(pq_inputs, max_workers=8):
    """Run postprocess + evaluate_image_official in parallel across images.

    Args:
        pq_inputs: list of (np_prob, hv, nc_logits, mask, inst_gt,
                            np_thresh, min_area, energy_thresh,
                            sobel_ksize, marker_ksize, match_iou)
        max_workers: number of parallel processes

    Returns:
        list of per-image result dicts (same order as inputs)
    """
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_eval_one_pq, pq_inputs))


# ==============================================================================
#  Final evaluation (shared by train.py and kd_train.py)
# ==============================================================================

def evaluate_final(model, loader, device, data_root, val_fold,
                   eval_np_thresh=0.5, eval_min_area=10,
                   eval_energy_thresh=0.3, eval_sobel_ksize=21,
                   eval_marker_ksize=3, eval_match_iou=0.5):
    """Run full PQ evaluation on the validation set and print + return metrics."""
    import torch

    model.eval()
    pq_inputs = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)

            np_prob = torch.sigmoid(outputs["np"]).cpu().numpy()
            hv = outputs["hv"].cpu().numpy()
            nc_logits = outputs["nc"].cpu().numpy()
            masks = batch["mask"].cpu().numpy()
            inst_gts = batch["inst_gt"].cpu().numpy()

            for b in range(images.shape[0]):
                pq_inputs.append((
                    np_prob[b, 0], hv[b], nc_logits[b],
                    masks[b], inst_gts[b],
                    eval_np_thresh, eval_min_area,
                    eval_energy_thresh, eval_sobel_ksize,
                    eval_marker_ksize, eval_match_iou,
                ))

    results = evaluate_parallel(pq_inputs)
    tissue = load_tissue_types(data_root, val_fold)
    metrics = aggregate_metrics(results, tissue_types=tissue)

    print(f"\n{'='*60}")
    print(f"Final PQ Evaluation (official PanNuke protocol)")
    print(f"  Instances: GT={metrics['total_gt_instances']}  Pred={metrics['total_pred_instances']}")
    for i, name in enumerate(CLASS_NAMES):
        gt_c = metrics['gt_per_class'][i]
        pd_c = metrics['pred_per_class'][i]
        print(f"    {name:<14}: GT={gt_c:5d}  Pred={pd_c:5d}")
    print(f"  Binary:  DQb={metrics['DQb']:.4f}  SQb={metrics['SQb']:.4f}  bPQ={metrics['bPQ']:.4f}  bPQ_Tiss={metrics.get('bPQ_Tiss','N/A')}")
    print(f"  Multi:   DQm={metrics['DQm']:.4f}  SQm={metrics['SQm']:.4f}  mPQ={metrics['mPQ']:.4f}  mPQ_Tiss={metrics.get('mPQ_Tiss','N/A')}")
    for name, v in metrics['per_class'].items():
        print(f"    {name:<14}: DQ={v['DQ']:.4f}  SQ={v['SQ']:.4f}  PQ={v['PQ']:.4f}")
    return metrics
