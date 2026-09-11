import numpy as np

def outcomes(visible, overlap):
    visible = np.asarray(visible)
    overlap = np.asarray(overlap, float)
    if visible.shape != overlap.shape or not np.isin(visible, [0, 1]).all():
        raise ValueError('visibility must contain aligned binary labels; unknown is not visible')
    if not np.isfinite(overlap).all():
        raise ValueError('fixed-candidate overlap is missing or non-finite')
    return np.where(visible == 0, 2, np.where(overlap >= 0.5, 0, 1)).astype(np.int8)

def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = np.maximum(a[..., :2], b[..., :2])
    hi = np.minimum(a[..., :2] + a[..., 2:], b[..., :2] + b[..., 2:])
    wh = np.maximum(hi-lo, 0)
    intersection = wh[..., 0] * wh[..., 1]
    union = a[..., 2]*a[..., 3] + b[..., 2]*b[..., 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
