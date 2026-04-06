"""
Worm Tracker - Revamped using MWT (Multi-Worm Tracker) algorithms
Based on: Buckingham & Bhatt et al. - Multi-Worm Tracker image processing methodology

Key algorithmic changes from prior version:
  1. Contour-based spine (11-point) instead of skeletonization
  2. MWT end-detection: furthest-pair on resampled contour for head/tail
  3. Dual-threshold flood-fill segmentation (T and beta_T * T)
  4. Decaying average background estimate: B(t+1) = (1-alpha)*B(t) + alpha*I(t), alpha = 2^-5
  5. Overlap-based object persistence (>=50% pixel overlap) instead of centroid jump distance
  6. Stringent/relaxed size threshold pairs for capture vs. tracking

IMPROVEMENTS:
  - CLAHE-based per-well illumination correction
  - Setup preview: worm pixels highlighted in colour with small-object filter slider
  - All trackbar labels drawn as image overlays (fixes Qt font issue)
  - Multi-object tracking: ALL blobs per well are tracked independently.
    Each blob gets a unique ObjectID (well-scoped, monotonically increasing).
    ObjectID is consistent across frames as long as >=50% pixel overlap holds.
    A gap of >=MISS_STREAK_RESET frames ends the track; if the object reappears
    it receives a new ObjectID.
    The CSV now has an ObjectID column; multiple rows per (frame, well) are
    normal — one per detected object.  Filter noise post-hoc by discarding
    short-lived ObjectIDs.

CSV columns — every raw primitive needed to derive all debris-filtering features:
  Tracking identity : Rel_Frame, TimeID, Well, ObjectID (NaN = no detection)
  Body geometry     : Angle, Area, Net_Spine_Len, Spine_Arc_Len,
                      Width_Mean, Width_Std, Width_CV, Width_Taper_Ratio, Width_Max,
                      Spine_Curvature_Total, Spine_Curvature_Std
  Contour/boundary  : Perimeter, Circularity, Solidity, Convex_Hull_Area,
                      Eccentricity, Ellipse_Major, Ellipse_Minor, Ellipse_Angle,
                      Bounding_Rect_AR, Boundary_Curv_Std,
                      Tip_Angle_Head, Tip_Angle_Tail
  Position          : Centroid_X, Centroid_Y, Head_X, Head_Y, Tail_X, Tail_Y

HOTKEYS (Setup): [j] Jump Frame | [SPACE] Start Analysis | [q] Quit
HOTKEYS (Analysis): [q] Quit and save
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
import math
import sys

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Video path is passed as a CLI argument:
#   python PySWIP_v4.py "worm videos/my_video.avi"

BG_ALPHA      = 2 ** -5        # decaying background alpha = 0.03125
BETA_T        = 0.8            # relaxed threshold = BETA_T * T
BETA_S        = 0.7            # relaxed min-area  = BETA_S * S_min

FPS            = 30
TARGET_WINDOWS = [(int(i * 60 * FPS), int((i + 1) * 60 * FPS)) for i in range(0, 91, 10)]

N_SPINE_POINTS     = 11
CONTOUR_RESAMPLE_N = 120

AREA_DROP_FRACTION  = 0.35   # blob area < fraction * EMA → noise, skip
AREA_EMA_ALPHA      = 0.2
MISS_STREAK_RESET   = 15     # consecutive missed frames before a track dies
                             # (raised from 5 — short-term bg noise was killing valid tracks)
MAX_ENDPOINT_TRAVEL = 40     # px — max tip jump before orientation is locked
MIN_TIP_SEPARATION  = 10     # px — tips closer than this → worm coiled, lock

# --- ObjectID inflation fix ---
CONFIRM_FRAMES   = 3   # frames a new blob must survive before getting an ObjectID
                       # and writing data rows.  Eliminates single-frame noise IDs.

# --- Background freeze grace period (prevents background from learning stationary objects
#     and then re-detecting them as new blobs every ~32 frames) ---
N_GHOST_FRAMES   = 60  # keep freeze mask for this many frames after a track dies

# --- Initial background blur for paralyzed-worm-from-frame-0 detection ---
# The reference frame bg is blurred with a kernel large enough to average out the
# thin worm body (~10px) but preserve the illumination field (~well diameter).
# Effective body width in px / well diameter in px ≈ 10/160 → kernel must be >> 10px.
BG_INIT_BLUR_K   = 0   # 0 = auto: max(51, radius // 2 * 2 + 1).  Set manually to override.


# ---------------------------------------------------------------------------
# TRACK  –  per-object state bundle
# ---------------------------------------------------------------------------
class Track:
    """All per-object state for one tracked blob in one well."""

    _well_counters: dict = {}

    def __init__(self, wid: str, blob_mask: np.ndarray, area: float):
        # ObjectID is NOT assigned yet — only assigned after CONFIRM_FRAMES
        self._wid          = wid
        self.object_id     = None          # None until confirmed
        self.prev_mask     = blob_mask
        self.area_ema      = area
        self.miss_streak   = 0
        self.confirm_count = 1             # frames seen so far (starts at 1 for spawn frame)
        self.prev_head_pt  = None
        self.prev_tail_pt  = None

    def confirm(self):
        """Called when confirm_count reaches CONFIRM_FRAMES. Assigns a real ObjectID."""
        Track._well_counters[self._wid] = Track._well_counters.get(self._wid, 0) + 1
        self.object_id = Track._well_counters[self._wid]

    @property
    def is_confirmed(self):
        return self.object_id is not None

    @classmethod
    def reset_well_counter(cls, wid: str):
        cls._well_counters[wid] = 0


# ---------------------------------------------------------------------------
# ILLUMINATION CORRECTION
# ---------------------------------------------------------------------------
def correct_illumination(roi_gray):
    """
    Stage 1 – background division (kernel = ¼ of smaller dim, min 51 px, odd).
    Stage 2 – CLAHE for local contrast equalisation.
    """
    if roi_gray is None or roi_gray.size == 0:
        return roi_gray
    h, w  = roi_gray.shape
    k     = max(51, (min(h, w) // 4) | 1)
    bg    = np.maximum(cv2.GaussianBlur(roi_gray, (k, k), 0).astype(np.float32), 1.0)
    corr  = np.clip((roi_gray.astype(np.float32) / bg) * float(np.mean(bg)),
                    0, 255).astype(np.uint8)
    tile  = max(4, min(h, w) // 16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    return clahe.apply(corr)


# ---------------------------------------------------------------------------
# MWT DUAL-THRESHOLD FLOOD-FILL SEGMENTATION
# ---------------------------------------------------------------------------
def dual_threshold_segment(diff_img, mask, T, beta_T=BETA_T,
                            min_area=100, max_area=None):
    relaxed_T    = max(1, int(beta_T * T))
    seed_map     = (diff_img >= T)         & (mask > 0)
    relaxed_map  = (diff_img >= relaxed_T) & (mask > 0)
    seed_lbl     = cv2.connectedComponents(seed_map.astype(np.uint8),    connectivity=8)[1]
    relax_lbl    = cv2.connectedComponents(relaxed_map.astype(np.uint8), connectivity=8)[1]
    blobs, seen  = [], set()
    max_a        = np.inf if max_area is None else max_area
    for sid in range(1, seed_lbl.max() + 1):
        px = np.where(seed_lbl == sid)
        if len(px[0]) == 0:
            continue
        for rid in set(relax_lbl[px].tolist()) - {0}:
            if rid in seen:
                continue
            seen.add(rid)
            blob = (relax_lbl == rid).astype(np.uint8)
            area = int(blob.sum())
            if min_area <= area <= max_a:
                blobs.append(blob)
    return blobs


# ---------------------------------------------------------------------------
# CONTOUR / SPINE UTILITIES
# ---------------------------------------------------------------------------
def resample_contour(contour, n=CONTOUR_RESAMPLE_N):
    pts  = contour[:, 0, :].astype(np.float64)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    segs = np.append(segs, np.linalg.norm(pts[-1] - pts[0]))
    arc  = np.concatenate([[0], np.cumsum(segs)])
    if arc[-1] < 1e-9:
        return None
    t   = np.linspace(0, arc[-1], n, endpoint=False)
    xs  = np.interp(t, arc, np.append(pts[:, 0], pts[0, 0]))
    ys  = np.interp(t, arc, np.append(pts[:, 1], pts[0, 1]))
    pts2 = np.column_stack([xs, ys])
    from scipy.ndimage import gaussian_filter1d
    pts2[:, 0] = gaussian_filter1d(pts2[:, 0], sigma=2, mode='wrap')
    pts2[:, 1] = gaussian_filter1d(pts2[:, 1], sigma=2, mode='wrap')
    return pts2


def get_contour_endpoints(smooth_pts):
    if len(smooth_pts) < 12:
        return None, None
    c = smooth_pts.mean(axis=0)
    a = int(np.argmax(np.linalg.norm(smooth_pts - c, axis=1)))
    b = int(np.argmax(np.linalg.norm(smooth_pts - smooth_pts[a], axis=1)))
    return a, b


def build_spine(smooth_pts, head_idx, tail_idx, n_points=N_SPINE_POINTS):
    pts = smooth_pts.astype(np.float64)
    arc_a = pts[head_idx:tail_idx + 1] if head_idx <= tail_idx \
            else np.concatenate([pts[head_idx:], pts[:tail_idx + 1]])
    raw_b = pts[tail_idx:head_idx + 1] if tail_idx <= head_idx \
            else np.concatenate([pts[tail_idx:], pts[:head_idx + 1]])
    arc_b = raw_b[::-1]

    def sample_arc(path, k):
        if len(path) < 2:
            return np.tile(path[0], (k, 1)).astype(np.float64)
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        if d[-1] < 1e-9:
            return np.tile(path[0], (k, 1)).astype(np.float64)
        out = []
        for t in np.linspace(0, d[-1], k):
            i   = np.clip(np.searchsorted(d, t, side='right'), 1, len(d) - 1)
            seg = d[i] - d[i - 1]
            a   = 0.0 if seg < 1e-9 else (t - d[i - 1]) / seg
            out.append((1 - a) * path[i - 1] + a * path[i])
        return np.array(out, dtype=np.float64)

    spine = (sample_arc(arc_a, n_points) + sample_arc(arc_b, n_points)) / 2.0
    side_a = sample_arc(arc_a, n_points)
    side_b = sample_arc(arc_b, n_points)
    spine  = (side_a + side_b) / 2.0
    return spine, side_a, side_b


def get_bending_angle_from_spine(spine):
    if spine is None or len(spine) < 3:
        return None
    s  = np.asarray(spine, dtype=np.float64)
    v1 = s[0]  - s[len(s) // 2]
    v2 = s[-1] - s[len(s) // 2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))



# ---------------------------------------------------------------------------
# SHAPE FEATURE BUNDLE
# ---------------------------------------------------------------------------
# Named tuple / dict keys for all columns written to CSV
CSV_COLUMNS = [
    'Rel_Frame', 'TimeID', 'Well', 'ObjectID',
    # body geometry
    'Angle', 'Area', 'Net_Spine_Len', 'Spine_Arc_Len',
    'Width_Mean', 'Width_Std', 'Width_CV', 'Width_Taper_Ratio', 'Width_Max',
    'Spine_Curvature_Total', 'Spine_Curvature_Std',
    # contour / boundary
    'Perimeter', 'Circularity', 'Solidity', 'Convex_Hull_Area',
    'Eccentricity', 'Ellipse_Major', 'Ellipse_Minor', 'Ellipse_Angle',
    'Bounding_Rect_AR', 'Boundary_Curv_Std',
    'Tip_Angle_Head', 'Tip_Angle_Tail',
    # position
    'Centroid_X', 'Centroid_Y',
    'Head_X', 'Head_Y', 'Tail_X', 'Tail_Y',
]

_NAN_ROW_TAIL = [np.nan] * (len(CSV_COLUMNS) - 4)   # everything after the 4 identity cols


def _tip_angle(smooth_pts, tip_idx, step=6):
    """
    Interior angle of the contour at tip_idx.
    Computed between the two tangent vectors leaving the tip at ±step indices.
    Returns degrees; 0 = perfectly sharp, 180 = flat/blunt.
    """
    n  = len(smooth_pts)
    p  = smooth_pts[tip_idx]
    pa = smooth_pts[(tip_idx - step) % n]
    pb = smooth_pts[(tip_idx + step) % n]
    v1 = pa - p
    v2 = pb - p
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


def compute_shape_features(best_cnt, smooth_pts, spine, side_a, side_b,
                            head_idx, tail_idx, angle, blob_area):
    """
    Compute every shape descriptor from already-available objects.
    Returns a dict keyed by CSV column name (only the non-identity columns).
    All values are plain Python floats or np.nan.

    Parameters
    ----------
    best_cnt   : raw OpenCV contour  (N,1,2) int32
    smooth_pts : resampled contour   (120,2) float64
    spine      : midpoint spine      (N_SPINE_POINTS,2) float64
    side_a     : one outline half    (N_SPINE_POINTS,2) float64
    side_b     : other outline half  (N_SPINE_POINTS,2) float64
    head_idx   : index into smooth_pts of head tip
    tail_idx   : index into smooth_pts of tail tip
    angle      : 3-point bending angle (already computed)
    blob_area  : float pixel area
    """
    feat = {}

    # ------------------------------------------------------------------ #
    # 1.  Body geometry from spine + side arrays                          #
    # ------------------------------------------------------------------ #
    feat['Angle'] = round(angle, 4) if angle is not None else np.nan
    feat['Area']  = float(blob_area)

    # Net (straight-line) spine length
    net_len = float(np.linalg.norm(spine[0] - spine[-1]))
    feat['Net_Spine_Len'] = round(net_len, 4)

    # True arc length along spine
    arc_len = float(np.sum(np.linalg.norm(np.diff(spine, axis=0), axis=1)))
    feat['Spine_Arc_Len'] = round(arc_len, 4)

    # Width profile: distance between corresponding side_a / side_b points
    widths = np.linalg.norm(side_a - side_b, axis=1)   # shape (N_SPINE_POINTS,)
    w_mean = float(np.mean(widths))
    w_std  = float(np.std(widths))
    w_max  = float(np.max(widths))
    feat['Width_Mean']        = round(w_mean, 4)
    feat['Width_Std']         = round(w_std,  4)
    feat['Width_CV']          = round(w_std / w_mean, 4) if w_mean > 0 else np.nan
    # Taper ratio: (tip widths) / max width.  Low → pointed ends (worm-like)
    feat['Width_Taper_Ratio'] = round((widths[0] + widths[-1]) / (2.0 * w_max), 4) \
                                if w_max > 0 else np.nan
    feat['Width_Max']         = round(w_max, 4)

    # Spine curvature: turning angles between consecutive segments
    segs = np.diff(spine, axis=0)                        # (N-1, 2)
    seg_angles = np.arctan2(segs[:, 1], segs[:, 0])     # (N-1,)
    turning    = np.diff(seg_angles)                     # (N-2,)
    # Wrap to [-π, π]
    turning = (turning + np.pi) % (2 * np.pi) - np.pi
    feat['Spine_Curvature_Total'] = round(float(np.sum(np.abs(turning))) * 180 / np.pi, 4)
    feat['Spine_Curvature_Std']   = round(float(np.std(turning))         * 180 / np.pi, 4)

    # ------------------------------------------------------------------ #
    # 2.  Contour / boundary features                                     #
    # ------------------------------------------------------------------ #
    perimeter = cv2.arcLength(best_cnt, closed=True)
    feat['Perimeter'] = round(float(perimeter), 4)

    # Circularity: 1 = perfect circle, lower = elongated / irregular
    feat['Circularity'] = round(4 * math.pi * blob_area / (perimeter ** 2), 4) \
                          if perimeter > 0 else np.nan

    # Convex hull and solidity
    hull      = cv2.convexHull(best_cnt)
    hull_area = float(cv2.contourArea(hull))
    feat['Convex_Hull_Area'] = round(hull_area, 4)
    feat['Solidity']         = round(blob_area / hull_area, 4) if hull_area > 0 else np.nan

    # Fitted ellipse (needs ≥ 5 points)
    if len(best_cnt) >= 5:
        try:
            (ex, ey), (ma, Mi), ea = cv2.fitEllipse(best_cnt)
            major = max(ma, Mi)
            minor = min(ma, Mi)
            feat['Ellipse_Major'] = round(float(major), 4)
            feat['Ellipse_Minor'] = round(float(minor), 4)
            feat['Ellipse_Angle'] = round(float(ea), 4)
            # Eccentricity from semi-axes a, b where a ≥ b
            a, b  = major / 2.0, minor / 2.0
            feat['Eccentricity'] = round(float(np.sqrt(1 - (b / a) ** 2)), 4) \
                                   if a > 0 else np.nan
        except cv2.error:
            feat['Ellipse_Major'] = feat['Ellipse_Minor'] = \
            feat['Ellipse_Angle'] = feat['Eccentricity'] = np.nan
    else:
        feat['Ellipse_Major'] = feat['Ellipse_Minor'] = \
        feat['Ellipse_Angle'] = feat['Eccentricity'] = np.nan

    # Minimum area bounding rectangle aspect ratio
    _, (rw, rh), _ = cv2.minAreaRect(best_cnt)
    long_side  = max(rw, rh)
    short_side = min(rw, rh)
    feat['Bounding_Rect_AR'] = round(long_side / short_side, 4) \
                                if short_side > 0 else np.nan

    # Boundary curvature std from resampled contour tangent angles
    tangents = np.diff(smooth_pts, axis=0)
    tang_all = np.vstack([tangents, smooth_pts[0] - smooth_pts[-1]])
    tang_ang = np.arctan2(tang_all[:, 1], tang_all[:, 0])
    curv     = np.diff(np.unwrap(tang_ang))
    feat['Boundary_Curv_Std'] = round(float(np.std(curv)), 6)

    # Tip sharpness angles
    feat['Tip_Angle_Head'] = round(_tip_angle(smooth_pts, head_idx), 4) \
                             if head_idx is not None else np.nan
    feat['Tip_Angle_Tail'] = round(_tip_angle(smooth_pts, tail_idx), 4) \
                             if tail_idx is not None else np.nan

    # ------------------------------------------------------------------ #
    # 3.  Position                                                        #
    # ------------------------------------------------------------------ #
    M = cv2.moments(best_cnt)
    if M['m00'] > 0:
        feat['Centroid_X'] = round(float(M['m10'] / M['m00']), 3)
        feat['Centroid_Y'] = round(float(M['m01'] / M['m00']), 3)
    else:
        feat['Centroid_X'] = feat['Centroid_Y'] = np.nan

    feat['Head_X'] = round(float(smooth_pts[head_idx][0]), 3) if head_idx is not None else np.nan
    feat['Head_Y'] = round(float(smooth_pts[head_idx][1]), 3) if head_idx is not None else np.nan
    feat['Tail_X'] = round(float(smooth_pts[tail_idx][0]), 3) if tail_idx is not None else np.nan
    feat['Tail_Y'] = round(float(smooth_pts[tail_idx][1]), 3) if tail_idx is not None else np.nan

    return feat


def compute_overlap_fraction(mask_a, mask_b):
    """Overlap pixels / area of mask_a."""
    if mask_a is None or mask_b is None:
        return 0.0
    a = float(mask_a.sum())
    return 0.0 if a == 0 else float(np.logical_and(mask_a, mask_b).sum()) / a


# ---------------------------------------------------------------------------
# MULTI-OBJECT MATCHING
# ---------------------------------------------------------------------------
def match_blobs_to_tracks(blobs: list, tracks: list, min_overlap: float = 0.5):
    """
    Greedy one-to-one assignment of blobs to existing tracks by overlap.

    Returns
    -------
    matched          : list[(track, blob)]
    unmatched_blobs  : blobs with no matching track  → spawn new Track
    unmatched_tracks : tracks with no matching blob   → increment miss streak
    """
    if not blobs or not tracks:
        return [], list(blobs), list(tracks)

    ov = np.zeros((len(tracks), len(blobs)), dtype=np.float32)
    for ti, trk in enumerate(tracks):
        for bi, blob in enumerate(blobs):
            if trk.prev_mask is not None and trk.prev_mask.shape == blob.shape:
                ov[ti, bi] = compute_overlap_fraction(blob, trk.prev_mask)

    matched, used_t, used_b = [], set(), set()
    while True:
        ti, bi = np.unravel_index(np.argmax(ov), ov.shape)
        if ov[ti, bi] < min_overlap:
            break
        matched.append((tracks[ti], blobs[bi]))
        used_t.add(ti); used_b.add(bi)
        ov[ti, :] = -1; ov[:, bi] = -1

    return (matched,
            [b for i, b in enumerate(blobs)  if i not in used_b],
            [t for i, t in enumerate(tracks) if i not in used_t])


# ---------------------------------------------------------------------------
# SPINE EXTRACTION FOR ONE BLOB
# ---------------------------------------------------------------------------
def extract_spine_data(best_cnt, trk: Track):
    """
    Compute spine + all geometric primitives for a confirmed contour.
    Applies temporal orientation consistency against trk.prev_head_pt/tail_pt.

    Returns
    -------
    (spine, side_a, side_b, smooth_pts, head_idx, tail_idx,
     angle, net_spine_len, conf_head, conf_tail)
    or a 10-tuple of None on failure.
    """
    _fail = (None,) * 10
    if best_cnt is None or len(best_cnt) < 12:
        return _fail
    smooth_pts = resample_contour(best_cnt)
    if smooth_pts is None:
        return _fail
    head_idx, tail_idx = get_contour_endpoints(smooth_pts)
    if head_idx is None:
        return _fail

    cand_h = smooth_pts[head_idx]
    cand_t = smooth_pts[tail_idx]
    prev_h, prev_t = trk.prev_head_pt, trk.prev_tail_pt

    if prev_h is not None and prev_t is not None:
        cost_same = (np.linalg.norm(cand_h - prev_h) + np.linalg.norm(cand_t - prev_t))
        cost_flip = (np.linalg.norm(cand_h - prev_t) + np.linalg.norm(cand_t - prev_h))
        if cost_flip < cost_same:
            head_idx, tail_idx = tail_idx, head_idx
            cand_h, cand_t     = cand_t, cand_h
            best_move = min(np.linalg.norm(cand_h - prev_t),
                            np.linalg.norm(cand_t - prev_h))
        else:
            best_move = min(np.linalg.norm(cand_h - prev_h),
                            np.linalg.norm(cand_t - prev_t))
        tip_sep = np.linalg.norm(cand_h - cand_t)
        if tip_sep < MIN_TIP_SEPARATION or best_move > MAX_ENDPOINT_TRAVEL:
            if cost_flip < cost_same:
                head_idx, tail_idx = tail_idx, head_idx
            cand_h, cand_t = prev_h, prev_t

    spine, side_a, side_b = build_spine(smooth_pts, head_idx, tail_idx, N_SPINE_POINTS)
    angle   = get_bending_angle_from_spine(spine)
    if angle is None:
        return _fail
    net_len = math.dist(spine[0], spine[-1])
    return spine, side_a, side_b, smooth_pts, head_idx, tail_idx, angle, net_len, cand_h, cand_t


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def nothing(_):
    pass


# ---------------------------------------------------------------------------
# LABEL PANEL  (Qt font workaround — draw with cv2.putText only)
# ---------------------------------------------------------------------------
SLIDER_LABELS = [
    "Rows         : number of well rows in the grid",
    "Cols         : number of well columns in the grid",
    "Radius       : well circle radius (px, original scale)",
    "Start X      : X position of top-left well centre",
    "Start Y      : Y position of top-left well centre",
    "Spacing X    : horizontal distance between well centres",
    "Spacing Y    : vertical distance between well centres",
    "Adjust Mode  : 0=grid click  1=drag individual wells",
    "Ref Frame    : reference video frame for preview",
    "Threshold    : detection sensitivity  (higher = less sensitive)",
    "Small Obj    : minimum blob area to keep  (filters noise, px^2)",
    "Max Area     : maximum blob area to keep  (px^2)",
    "FAST MODE    : 0=slow+visualise  1=fast (no display)",
]


def build_label_panel(width=520):
    lh, pad = 26, 8
    h   = len(SLIDER_LABELS) * lh + pad * 2 + 30
    img = np.full((h, width, 3), (25, 25, 35), dtype=np.uint8)
    cv2.putText(img, "SLIDER REFERENCE", (pad, pad + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(img, (pad, pad + 22), (width - pad, pad + 22), (0, 180, 200), 1)
    for i, label in enumerate(SLIDER_LABELS):
        col = (180, 255, 160) if i % 2 == 0 else (160, 220, 255)
        cv2.putText(img, label, (pad, pad + 30 + i * lh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# WELL MANAGEMENT
# ---------------------------------------------------------------------------
well_data     = {}
current_stats = {}
dragging_wid  = None


def add_well(wid, cx, cy, r, start_frame, cap, h_orig, w_orig):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, bg_frame = cap.read()
    if not ret:
        return
    y1, y2 = max(0, cy - r), min(h_orig, cy + r)
    x1, x2 = max(0, cx - r), min(w_orig, cx + r)
    gray   = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    roi_bg = gray[y1:y2, x1:x2]
    if roi_bg.size == 0:
        return

    roi_corr = correct_illumination(roi_bg)

    # -----------------------------------------------------------------------
    # PARALYZED-WORM-FROM-FRAME-0 FIX
    # Build the initial background by blurring the corrected reference frame
    # with a kernel large enough to average out thin worm bodies (~10px wide)
    # while preserving the broader illumination field.
    # A worm is ~5-10% of the well diameter in width; kernel = ~25% of diameter
    # ensures it spans several worm widths and will blur them out entirely.
    # -----------------------------------------------------------------------
    roi_h, roi_w = roi_corr.shape
    k = BG_INIT_BLUR_K if BG_INIT_BLUR_K > 0 \
        else max(51, (min(roi_h, roi_w) // 4) | 1)
    roi_bg_f = cv2.GaussianBlur(roi_corr, (k, k), 0).astype(np.float32)
    # Do NOT apply the small (3,3) blur on top — the large blur IS the background.

    Track.reset_well_counter(wid)

    well_data[wid] = {
        'bg_float'        : roi_bg_f.copy(),
        'start_f'         : start_frame,
        'coords'          : (x1, y1, x2, y2),
        'center'          : (cx, cy),
        'radius'          : r,
        'last_time_id'    : None,
        'tracks'          : [],          # list[Track]
        # Ghost-freeze: accumulate last-known masks of recently-dead tracks.
        # Keys = frame_number_of_death, values = binary mask.
        # Prevents background from learning stationary objects and re-firing.
        'ghost_masks'     : {},          # {death_frame: mask}
    }


# ---------------------------------------------------------------------------
# CLICK / DRAG EVENT
# ---------------------------------------------------------------------------
def make_click_event(cap, h_orig, w_orig, total_f):
    def click_event(event, x, y, flags, param):
        global well_data, dragging_wid
        scale  = w_orig / 1000
        rx, ry = int(x * scale), int(y * scale)
        r           = cv2.getTrackbarPos('Radius',      'Setup')
        adjust_mode = cv2.getTrackbarPos('Adjust Mode', 'Setup')
        well_start  = cv2.getTrackbarPos('Ref Frame',   'Setup')

        if adjust_mode == 0:
            if event == cv2.EVENT_LBUTTONDOWN:
                sx   = cv2.getTrackbarPos('Start X',   'Setup')
                sy   = cv2.getTrackbarPos('Start Y',   'Setup')
                dx   = cv2.getTrackbarPos('Spacing X', 'Setup')
                dy   = cv2.getTrackbarPos('Spacing Y', 'Setup')
                rows = cv2.getTrackbarPos('Rows',      'Setup')
                cols = cv2.getTrackbarPos('Cols',      'Setup')
                for row in range(rows):
                    for col in range(cols):
                        cx2, cy2 = sx + col * dx, sy + row * dy
                        if np.hypot(rx - cx2, ry - cy2) < r:
                            wid = f"R{row}C{col}"
                            if wid in well_data:
                                del well_data[wid]
                            else:
                                add_well(wid, cx2, cy2, r, well_start, cap, h_orig, w_orig)
                            return
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                for wid, info in well_data.items():
                    cx2, cy2 = info['center']
                    if np.hypot(rx - cx2, ry - cy2) < r:
                        dragging_wid = wid
                        break
            elif event == cv2.EVENT_MOUSEMOVE and dragging_wid is not None:
                well_data[dragging_wid]['center'] = (rx, ry)
            elif event == cv2.EVENT_LBUTTONUP and dragging_wid is not None:
                cx2, cy2 = well_data[dragging_wid]['center']
                add_well(dragging_wid, cx2, cy2, r, well_start, cap, h_orig, w_orig)
                dragging_wid = None

    return click_event


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
def draw_dashboard(rows, cols, well_data, current_stats):
    cell_w, cell_h = 260, 120
    dash = np.zeros((rows * cell_h + 30, cols * cell_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            wid  = f"R{r}C{c}"
            x, y = c * cell_w, r * cell_h
            if wid in well_data:
                stat = current_stats.get(wid, {"status": "WAITING", "objects": []})
                cv2.rectangle(dash, (x+2, y+2), (x+cell_w-4, y+cell_h-4), (40, 40, 40), -1)
                cv2.putText(dash, f"{wid}: {stat['status']}", (x+8, y+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                objs = stat.get('objects', [])
                cv2.putText(dash, f"Active objects: {len(objs)}", (x+8, y+38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
                for oi, obj in enumerate(objs[:3]):
                    ang_s  = f"{obj['angle']:.0f}d" if obj.get('angle') is not None else "NaN"
                    area_s = str(int(obj['area']))   if obj.get('area')  is not None else "NaN"
                    cv2.putText(dash, f"  OBJ{obj['id']}: {ang_s}  area={area_s}",
                                (x+8, y+55 + oi * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 255, 160), 1)
            else:
                cv2.rectangle(dash, (x+2, y+2), (x+cell_w-4, y+cell_h-4), (20, 20, 20), -1)
    return dash


# ---------------------------------------------------------------------------
# SETUP PREVIEW
# ---------------------------------------------------------------------------
def build_worm_preview(canvas, gray_orig, scale_d, T_val, small_obj_area,
                        well_data, rows, cols, r, sx, sy, dx, dy, adj):
    h_f, w_f = gray_orig.shape[:2]
    preview  = canvas.copy()
    rois = {}
    if adj == 0:
        for row in range(rows):
            for col in range(cols):
                rois[f"R{row}C{col}"] = (sx + col * dx, sy + row * dy)
    for wid, info in well_data.items():
        rois[wid] = info['center']

    for wid, (cx_r, cy_r) in rois.items():
        x1, x2 = max(0, cx_r - r), min(w_f, cx_r + r)
        y1, y2 = max(0, cy_r - r), min(h_f, cy_r + r)
        roi    = gray_orig[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_c = correct_illumination(roi)
        k     = max(51, (min(roi_c.shape[0], roi_c.shape[1]) // 4) | 1)
        bg    = np.maximum(cv2.GaussianBlur(roi_c, (k, k), 0).astype(np.float32), 1.0)
        diff  = cv2.GaussianBlur(
            np.clip(cv2.absdiff(roi_c.astype(np.float32), bg), 0, 255).astype(np.uint8),
            (5, 5), 0)
        _, bin_img = cv2.threshold(diff, T_val, 255, cv2.THRESH_BINARY)
        roi_h, roi_w = roi_c.shape
        circ = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(circ, (roi_w // 2, roi_h // 2), r, 255, -1)
        bin_img = cv2.bitwise_and(bin_img, circ)
        if small_obj_area > 0:
            n_lbl, lbl_map, stats_, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
            filt = np.zeros_like(bin_img)
            for lbl in range(1, n_lbl):
                if stats_[lbl, cv2.CC_STAT_AREA] >= small_obj_area:
                    filt[lbl_map == lbl] = 255
            bin_img = filt
        if bin_img.sum() == 0:
            continue
        x1_d, y1_d = int(x1 * scale_d), int(y1 * scale_d)
        x2_d, y2_d = int(x2 * scale_d), int(y2 * scale_d)
        if x2_d <= x1_d or y2_d <= y1_d:
            continue
        bin_s = cv2.resize(bin_img, (x2_d - x1_d, y2_d - y1_d), interpolation=cv2.INTER_NEAREST)
        gmask = bin_s > 0
        roi_d = preview[y1_d:y2_d, x1_d:x2_d]
        if roi_d.shape[:2] != gmask.shape:
            continue
        roi_d[gmask] = (roi_d[gmask].astype(np.float32) * 0.35 +
                        np.array([0, 255, 80], dtype=np.float32) * 0.65).astype(np.uint8)
        preview[y1_d:y2_d, x1_d:x2_d] = roi_d
    return preview


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    global dragging_wid

    if len(sys.argv) != 2:
        print("Error: expected exactly one argument (path to video file).")
        print(f"Usage: python PySWIP_v4.py <video_path>")
        print(f"Example: python PySWIP_v4.py \"worm videos/test.avi\"")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: video not found at '{video_path}'")
        sys.exit(1)

    # Derive output directory from video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join('outputs', video_name)
    os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.join(output_dir, 'setup_config.json')
    csv_file    = os.path.join(output_dir, 'worm_results.csv')

    cap     = cv2.VideoCapture(video_path)
    h_orig  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use local config resolver instead of global CONFIG_FILE
    def load_config():
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    config  = load_config()

    # -----------------------------------------------------------------------
    # SETUP WINDOW
    # -----------------------------------------------------------------------
    cv2.namedWindow('Setup',  cv2.WINDOW_NORMAL)
    cv2.namedWindow('Labels', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Setup', make_click_event(cap, h_orig, w_orig, total_f))

    label_img = build_label_panel()
    cv2.imshow('Labels', label_img)
    cv2.resizeWindow('Labels', 520, label_img.shape[0])

    default_params = {
        'Rows': 3, 'Cols': 4, 'Radius': 80,
        'Start X': 100, 'Start Y': 100,
        'Spacing X': 200, 'Spacing Y': 200,
    }
    for k, v in default_params.items():
        max_v = w_orig if 'X' in k else h_orig
        cv2.createTrackbar(k, 'Setup', config.get(k, v), max_v, nothing)

    cv2.createTrackbar('Adjust Mode', 'Setup', 0, 1, nothing)
    cv2.createTrackbar('Ref Frame',   'Setup', 0, total_f - 1, nothing)
    cv2.createTrackbar('Threshold',   'Setup', config.get('Threshold', 20), 100, nothing)
    cv2.createTrackbar('Small Obj',   'Setup', config.get('Small Obj', 50), 2000, nothing)
    cv2.createTrackbar('Max Area',    'Setup', config.get('Max Area', 3000), 20000, nothing)
    cv2.createTrackbar('FAST MODE',   'Setup', 0, 1, nothing)

    final_thresh = 20; final_small_obj = 50; final_max_area = 3000; final_fast_mode = 0

    print("--- SETUP MODE ---  [j] Jump  [SPACE] Start  [q] Quit")

    while True:
        current_f = cv2.getTrackbarPos('Ref Frame', 'Setup')
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_f)
        ret, frame = cap.read()
        if not ret:
            break

        canvas_h  = int(h_orig * (1000 / w_orig))
        canvas    = cv2.resize(frame, (1000, canvas_h))
        scale_d   = 1000 / w_orig
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        r     = cv2.getTrackbarPos('Radius',      'Setup')
        rows  = cv2.getTrackbarPos('Rows',        'Setup')
        cols  = cv2.getTrackbarPos('Cols',        'Setup')
        sx    = cv2.getTrackbarPos('Start X',     'Setup')
        sy    = cv2.getTrackbarPos('Start Y',     'Setup')
        dx    = cv2.getTrackbarPos('Spacing X',   'Setup')
        dy    = cv2.getTrackbarPos('Spacing Y',   'Setup')
        T_val = cv2.getTrackbarPos('Threshold',   'Setup')
        sobj  = cv2.getTrackbarPos('Small Obj',   'Setup')
        max_a = cv2.getTrackbarPos('Max Area',    'Setup')
        fast  = cv2.getTrackbarPos('FAST MODE',   'Setup')
        adj   = cv2.getTrackbarPos('Adjust Mode', 'Setup')
        final_thresh, final_small_obj, final_max_area, final_fast_mode = T_val, sobj, max_a, fast

        canvas = build_worm_preview(canvas, gray_full, scale_d, T_val, sobj,
                                    well_data, rows, cols, r, sx, sy, dx, dy, adj)

        cv2.putText(canvas, f"Frame: {current_f}/{total_f} | [j] Jump | SPACE to Start",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        mode_txt = "MODE: ADJUST (Drag)" if adj else "MODE: GRID (Click to Lock/Unlock)"
        cv2.putText(canvas, mode_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 165, 255) if adj else (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"Thresh={T_val}  SmallObj>={sobj}px  MaxArea<={max_a}px  "
                    "[GREEN=detected pixels]",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        for wid, info in well_data.items():
            cxw, cyw = info['center']
            cx_d, cy_d, cr_d = int(cxw*scale_d), int(cyw*scale_d), int(r*scale_d)
            color = (0, 255, 255) if wid == dragging_wid else (255, 80, 80)
            cv2.circle(canvas, (cx_d, cy_d), cr_d, color, 2)
            cv2.putText(canvas, wid, (cx_d-15, cy_d+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if adj == 0:
            for row in range(rows):
                for col in range(cols):
                    if f"R{row}C{col}" not in well_data:
                        cx_g = int((sx + col * dx) * scale_d)
                        cy_g = int((sy + row * dy) * scale_d)
                        cv2.circle(canvas, (cx_g, cy_g), int(r*scale_d), (0, 220, 0), 2)

        cv2.imshow('Setup', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('j'):
            try:
                v = int(input("Jump to frame: "))
                if 0 <= v < total_f:
                    cv2.setTrackbarPos('Ref Frame', 'Setup', v)
            except Exception:
                pass
        if key == ord(' '):
            sp = {k: cv2.getTrackbarPos(k, 'Setup') for k in default_params}
            sp['Threshold'] = T_val; sp['Small Obj'] = sobj; sp['Max Area'] = max_a
            saved_rows = cv2.getTrackbarPos('Rows', 'Setup')
            saved_cols = cv2.getTrackbarPos('Cols', 'Setup')
            with open(config_file, 'w') as f:
                json.dump(sp, f)
            break
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return

    cv2.destroyAllWindows()
    if not well_data:
        print("No wells selected. Exiting."); cap.release(); return

    row_indices = [int(k[1])              for k in well_data if k.startswith('R')]
    col_indices = [int(k[k.index('C')+1:]) for k in well_data if 'C' in k]
    dash_rows   = (max(row_indices) + 1) if row_indices else saved_rows
    dash_cols   = (max(col_indices) + 1) if col_indices else saved_cols

    # -----------------------------------------------------------------------
    # ANALYSIS LOOP
    # -----------------------------------------------------------------------
    for wid in well_data:
        current_stats[wid] = {"status": "WAITING", "objects": []}

    data_buffer    = []
    final_min_area = max(1, final_small_obj)
    global_start   = min(v['start_f'] for v in well_data.values())
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_start)

    # Precompute circle masks
    well_masks = {}
    for wid, info in well_data.items():
        x1, y1, x2, y2 = info['coords']
        roi_h, roi_w = y2 - y1, x2 - x1
        m = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(m, (roi_w // 2, roi_h // 2), info['radius'], 255, -1)
        well_masks[wid] = m

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print("--- ANALYSIS MODE ---  [q] Quit and save")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            curr_f    = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for wid, info in well_data.items():
                rel_frame = curr_f - info['start_f']
                if rel_frame < 0:
                    continue

                time_id = None
                for idx, (s, e) in enumerate(TARGET_WINDOWS):
                    if s <= rel_frame < e:
                        time_id = idx + 1; break

                # Reset all tracks on window boundary
                if time_id != info['last_time_id']:
                    info['tracks']       = []
                    info['ghost_masks']  = {}
                    info['last_time_id'] = time_id
                    Track.reset_well_counter(wid)

                if time_id is None:
                    current_stats[wid]["status"] = "IDLE"
                    continue
                current_stats[wid]["status"] = "REC"

                # ---- BACKGROUND UPDATE ----
                x1, y1, x2, y2 = info['coords']
                roi   = correct_illumination(gray_full[y1:y2, x1:x2].copy())
                roi_f = roi.astype(np.float32)

                # Freeze background under every ACTIVE track mask
                freeze = np.zeros(roi_f.shape, dtype=bool)
                for trk in info['tracks']:
                    if trk.prev_mask is not None and trk.prev_mask.shape == roi_f.shape:
                        freeze |= (trk.prev_mask > 0)

                # Also freeze under recently-dead track positions (ghost masks).
                # This prevents the background EMA from learning a stationary object
                # during the gap between track-death and track-respawn, which would
                # cause the diff to collapse → object invisible → new ID spawned → repeat.
                expired_deaths = [df for df in info['ghost_masks']
                                  if rel_frame - df > N_GHOST_FRAMES]
                for df in expired_deaths:
                    del info['ghost_masks'][df]
                for ghost_mask in info['ghost_masks'].values():
                    if ghost_mask.shape == roi_f.shape:
                        freeze |= (ghost_mask > 0)

                bg     = info['bg_float']
                new_bg = (1.0 - BG_ALPHA) * bg + BG_ALPHA * roi_f
                merged = bg.copy()
                merged[~freeze] = new_bg[~freeze]
                info['bg_float'] = merged

                diff_u8 = cv2.GaussianBlur(
                    np.clip(np.abs(roi_f - merged), 0, 255).astype(np.uint8),
                    (5, 5), 0)

                # ---- SEGMENTATION ----
                any_tracking = bool(info['tracks'])
                min_area_use = max(10, int(BETA_S * final_min_area)) if any_tracking \
                               else final_min_area
                max_area_use = int(final_max_area / BETA_S) if any_tracking \
                               else final_max_area

                blobs = dual_threshold_segment(
                    diff_u8, well_masks[wid],
                    T=final_thresh, beta_T=BETA_T,
                    min_area=min_area_use, max_area=max_area_use)

                # ---- MULTI-OBJECT MATCHING ----
                matched, new_blobs, lost_tracks = match_blobs_to_tracks(
                    blobs, info['tracks'], min_overlap=0.5)

                surviving = []
                frame_rows = []

                # A) Update matched tracks
                for trk, blob in matched:
                    blob_area = float(blob.sum())
                    # Reject if area collapsed (noise fragment matched by chance)
                    if blob_area < AREA_DROP_FRACTION * trk.area_ema:
                        trk.miss_streak += 1
                        if trk.miss_streak < MISS_STREAK_RESET:
                            surviving.append(trk)
                        else:
                            # Track dying — register ghost mask
                            if trk.prev_mask is not None:
                                info['ghost_masks'][rel_frame] = trk.prev_mask.copy()
                        continue

                    smoothed = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k_close)
                    cnts, _  = cv2.findContours(smoothed, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                    if not cnts:
                        trk.miss_streak += 1
                        if trk.miss_streak < MISS_STREAK_RESET:
                            surviving.append(trk)
                        else:
                            if trk.prev_mask is not None:
                                info['ghost_masks'][rel_frame] = trk.prev_mask.copy()
                        continue

                    best_cnt = max(cnts, key=cv2.contourArea)

                    # Advance confirmation counter for unconfirmed tracks
                    if not trk.is_confirmed:
                        trk.confirm_count += 1
                        if trk.confirm_count >= CONFIRM_FRAMES:
                            trk.confirm()   # assigns ObjectID now

                    result   = extract_spine_data(best_cnt, trk)
                    spine, side_a, side_b, smooth_pts, head_idx, tail_idx, \
                        angle, net_len, conf_h, conf_t = result

                    # Update track state regardless of whether spine succeeded
                    trk.area_ema    = AREA_EMA_ALPHA * blob_area + \
                                      (1 - AREA_EMA_ALPHA) * trk.area_ema
                    trk.miss_streak = 0
                    trk.prev_mask   = blob
                    if conf_h is not None:
                        trk.prev_head_pt = conf_h
                        trk.prev_tail_pt = conf_t
                    surviving.append(trk)

                    # Only write data rows for confirmed tracks
                    if not trk.is_confirmed:
                        continue

                    if angle is not None:
                        # Compute the full shape feature bundle
                        feat = compute_shape_features(
                            best_cnt, smooth_pts, spine, side_a, side_b,
                            head_idx, tail_idx, angle, blob_area)
                        row = [rel_frame, time_id, wid, trk.object_id] + \
                              [feat.get(c, np.nan) for c in CSV_COLUMNS[4:]]
                    else:
                        # Spine failed — still record identity + area + position from moments
                        M = cv2.moments(best_cnt)
                        cx = float(M['m10'] / M['m00']) if M['m00'] > 0 else np.nan
                        cy = float(M['m01'] / M['m00']) if M['m00'] > 0 else np.nan
                        row = list([rel_frame, time_id, wid, trk.object_id] + _NAN_ROW_TAIL)
                        _col = CSV_COLUMNS
                        row[_col.index('Area')]       = int(blob_area)
                        row[_col.index('Centroid_X')] = round(cx, 3)
                        row[_col.index('Centroid_Y')] = round(cy, 3)

                    frame_rows.append(row)

                    # Visualise
                    if not final_fast_mode and angle is not None:
                        cv2.drawContours(frame,
                            [best_cnt + np.array([[[x1, y1]]])], -1, (0, 200, 255), 1)
                        if spine is not None:
                            for si in range(len(spine) - 1):
                                cv2.line(frame,
                                    (int(spine[si][0]) + x1,   int(spine[si][1]) + y1),
                                    (int(spine[si+1][0]) + x1, int(spine[si+1][1]) + y1),
                                    (0, 255, 255), 1)
                            cv2.circle(frame,
                                (int(spine[0][0]) + x1,  int(spine[0][1]) + y1),
                                5, (255, 0, 0), -1)
                            cv2.circle(frame,
                                (int(spine[-1][0]) + x1, int(spine[-1][1]) + y1),
                                5, (0, 255, 0), -1)

                # B) Lost tracks — increment miss streak, keep if still alive
                for trk in lost_tracks:
                    trk.miss_streak += 1
                    if trk.miss_streak < MISS_STREAK_RESET:
                        surviving.append(trk)
                    else:
                        # Track dying — register last known mask as ghost
                        # so background stays frozen there for N_GHOST_FRAMES
                        if trk.prev_mask is not None:
                            info['ghost_masks'][rel_frame] = trk.prev_mask.copy()

                # C) Spawn candidate tracks for unmatched blobs.
                # No ObjectID yet — confirmation happens in section A over next frames.
                for blob in new_blobs:
                    new_trk = Track(wid, blob, float(blob.sum()))
                    surviving.append(new_trk)

                info['tracks'] = surviving

                # Write rows; if nothing detected emit a NaN sentinel row
                data_buffer.extend(frame_rows)
                if not frame_rows:
                    data_buffer.append(
                        [rel_frame, time_id, wid, np.nan] + list(_NAN_ROW_TAIL))

                # Dashboard — only show confirmed tracks with data rows
                obj_stats = []
                confirmed_surviving = [t for t in surviving if t.is_confirmed]
                for trk in confirmed_surviving:
                    row_match = next((r for r in frame_rows
                                      if r[3] == trk.object_id), None)
                    _ai = CSV_COLUMNS.index('Angle')
                    _ri = CSV_COLUMNS.index('Area')
                    obj_stats.append({
                        "id":    trk.object_id,
                        "angle": row_match[_ai] if row_match else None,
                        "area":  row_match[_ri] if row_match else None,
                    })
                current_stats[wid]["objects"] = obj_stats

            cv2.imshow('Live Dashboard',
                       draw_dashboard(dash_rows, dash_cols, well_data, current_stats))
            if not final_fast_mode:
                cv2.imshow('Slow Analysis',
                           cv2.resize(frame, (1000, int(h_orig * (1000 / w_orig)))))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if data_buffer:
            df = pd.DataFrame(data_buffer, columns=CSV_COLUMNS)
            df.to_csv(csv_file, index=False)
            valid = df.dropna(subset=['ObjectID'])
            print(f"Saved {len(df)} rows → {csv_file}")
            print(f"  {len(valid)} rows with detected objects, "
                  f"{int(valid['ObjectID'].nunique())} unique ObjectIDs")
            print(f"  Columns ({len(CSV_COLUMNS)}): {', '.join(CSV_COLUMNS)}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()