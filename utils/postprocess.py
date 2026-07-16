"""
HoVer-Net style post-processing for 3-head (NP + HV + NC) nuclei segmentation.

Pipeline:
    1. NP threshold → binary nucleus mask + CC filtering
    2. HV Sobel gradients → energy landscape
    3. Watershed with distance-transform seeds → instance map
    4. Per-instance classification via NC majority vote
"""
import cv2
import numpy as np
from typing import Tuple, Dict
from scipy.ndimage import measurements, binary_fill_holes


def _remove_small_objects(label_map: np.ndarray, min_size: int = 10) -> np.ndarray:
    if label_map.max() == 0:
        return label_map
    cleaned = np.zeros_like(label_map)
    for iid in np.unique(label_map):
        if iid == 0:
            continue
        mask = (label_map == iid).astype(np.uint8)
        if mask.sum() >= min_size:
            cleaned[label_map == iid] = iid
    return cleaned


def process_np_hv(
    np_map: np.ndarray,
    hv_map: np.ndarray,
    np_thresh: float = 0.5,
    min_area: int = 10,
    sobel_ksize: int = 21,
    energy_thresh: float = 0.4,
    marker_ksize: int = 5,
) -> np.ndarray:
    """
    HoVer-Net watershed instance segmentation from NP + HV maps.

    Args:
        np_map:        [H, W] float32, sigmoid nucleus probability
        hv_map:        [2, H, W] float32, predicted HV offsets
        np_thresh:     NP probability threshold
        min_area:      Minimum nucleus area (pixels)
        sobel_ksize:   Sobel kernel size for HV gradient (odd)
        energy_thresh: Energy landscape threshold (lower = more separation)
        marker_ksize:  Marker extraction kernel size

    Returns:
        inst_map: [H, W] int32 instance IDs (0=bg, 1..N)
    """
    # 1. NP threshold → CC filter
    blb = np.array(np_map >= np_thresh, dtype=np.int32)
    blb = measurements.label(blb)[0]
    blb = _remove_small_objects(blb, min_size=min_area)
    blb[blb > 0] = 1

    if blb.max() == 0:
        return np.zeros_like(blb, dtype=np.int32)

    # 2. Normalise HV → [0, 1]
    h_dir = cv2.normalize(hv_map[0], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(hv_map[1], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # 3. Sobel gradients → energy landscape
    sobel_h = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_v = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_h = 1.0 - cv2.normalize(sobel_h, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobel_v = 1.0 - cv2.normalize(sobel_v, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    energy = np.maximum(sobel_h, sobel_v)
    energy = energy - (1.0 - blb)
    energy = np.clip(energy, 0, None)

    # 4. Distance transform on inverted energy
    dist = (1.0 - energy) * blb
    dist = -cv2.GaussianBlur(dist.astype(np.float32), (3, 3), 0)

    # 5. Markers = blb minus high-energy ridges
    energy_mask = np.array(energy >= energy_thresh, dtype=np.int32)
    marker = blb - energy_mask
    marker = np.clip(marker, 0, None)
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marker_ksize, marker_ksize))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = _remove_small_objects(marker, min_size=min_area)

    if marker.max() == 0:
        return measurements.label(blb)[0].astype(np.int32)

    # 6. Watershed
    from skimage.segmentation import watershed
    inst_map = watershed(dist, markers=marker, mask=blb)
    return inst_map.astype(np.int32)


def classify_instances(
    inst_map: np.ndarray,
    nc_logits: np.ndarray,
    neo_conn_bias: float = 0.0,
) -> Dict[int, dict]:
    """
    Per-instance class assignment via per-pixel argmax + majority vote.

    Matches the official HoVer-Net post-processing: argmax per pixel,
    then count votes within each instance. If the most-voted class is
    background (0), the second-most-voted class is used (if available).

    Args:
        inst_map:  [H, W] int32 instance IDs
        nc_logits: [C, H, W] float32 raw logits (C = NUM_CLASSES, no bg channel)
        neo_conn_bias: if > 0, override connective→neoplastic when
            neoplastic_votes >= neo_conn_bias * connective_votes.
            e.g. 0.6 means "if neo votes ≥ 60% of conn votes, call it neo".

    Returns:
        {inst_id: {'type': int (0..C-1), 'type_prob': float}, ...}
    """
    if nc_logits.ndim == 2:
        pixel_types = nc_logits                   # precomputed argmax
    else:
        nc_logits = nc_logits.astype(np.float32)
        pixel_types = np.argmax(nc_logits, axis=0)  # [H, W], values 0..C-1

    inst_info = {}
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        mask = inst_map == iid
        inst_types = pixel_types[mask]  # class predictions within this instance

        # Majority vote
        type_ids, type_counts = np.unique(inst_types, return_counts=True)
        type_pairs = sorted(zip(type_ids, type_counts), key=lambda x: x[1], reverse=True)

        inst_type = int(type_pairs[0][0])
        type_dict = {int(t): int(c) for t, c in type_pairs}
        type_prob = type_dict[inst_type] / (mask.sum() + 1.0e-6)

        # Cost-sensitive: if connective(2) but neoplastic(0) is close, bias to neo
        if neo_conn_bias > 0 and inst_type == 2:
            neo_votes = type_dict.get(0, 0)
            conn_votes = type_dict.get(2, 0)
            if conn_votes > 0 and neo_votes >= neo_conn_bias * conn_votes:
                inst_type = 0
                type_prob = neo_votes / (mask.sum() + 1.0e-6)

        inst_info[int(iid)] = {
            'type': inst_type,
            'type_prob': float(type_prob),
        }
    return inst_info


def postprocess_nuclei(
    np_map: np.ndarray,
    hv_map: np.ndarray,
    nc_logits: np.ndarray,
    np_thresh: float = 0.5,
    min_area: int = 10,
    sobel_ksize: int = 21,
    energy_thresh: float = 0.4,
    marker_ksize: int = 5,
    neo_conn_bias: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full post-processing pipeline.

    Args:
        neo_conn_bias: if > 0, override connective→neoplastic when
            neoplastic pixel votes within instance are close to connective votes.

    Returns:
        inst_map:   [H, W] int32 instance IDs
        class_map:  [H, W] int32 class labels (0=bg, 1..C)
        inst_info:  {inst_id: {'type': ..., 'type_prob': ...}}
    """
    inst_map = process_np_hv(
        np_map, hv_map,
        np_thresh=np_thresh, min_area=min_area,
        sobel_ksize=sobel_ksize,
        energy_thresh=energy_thresh,
        marker_ksize=marker_ksize,
    )

    inst_info = classify_instances(inst_map, nc_logits, neo_conn_bias=neo_conn_bias)

    class_map = np.zeros_like(inst_map, dtype=np.int32)
    for iid, info in inst_info.items():
        class_map[inst_map == iid] = info['type'] + 1  # 1-indexed for vis

    return inst_map, class_map, inst_info
