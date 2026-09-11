"""Canonical Rejection-Evidence Packet (CREP) spatial canonicalizer.

This module only transforms already-computed current evidence and an unchanged
host candidate.  It does not decode a new candidate, modify native-valid
outputs, perform a second backbone forward, or make an admission decision.

Coordinate convention
---------------------
Response arrays have shape ``(height, width)``.  Point coordinates supplied to
the canonicalizer are zero-based response-cell-center coordinates: cell
``(row=y, column=x)`` has point coordinate ``(x, y)``.  Boxes are continuous
response-field edge coordinates ``(x0, y0, x1, y1)``.  TAPNext++ official
coordinates include a +0.5 offset; ``tapnext_evidence`` returns both the
official coordinate and the corresponding field-center coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
CandidateKind = Literal["point", "box"]


@dataclass(frozen=True)
class SpatialView:
    response: Array
    candidate_support: Array
    valid_support: Array


@dataclass(frozen=True)
class CREPPacket:
    global_view: SpatialView
    local_view: SpatialView
    native_margin: float
    candidate_kind: CandidateKind
    resolution: int
    rho: float


@dataclass(frozen=True)
class TAPNextEvidence:
    joint_field: Array
    candidate_official_xy: tuple[float, float]
    candidate_field_xy: tuple[float, float]
    probability_x: Array
    probability_y: Array


def peak_normalize(response: Array) -> Array:
    """Return non-negative response divided by its maximum.

    Negative inputs are rejected because CREP V1.1 defines a non-negative
    spatial evidence field before canonicalization.
    """

    value = np.asarray(response, dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("response must be a non-empty 2-D array")
    if not np.isfinite(value).all():
        raise ValueError("response contains non-finite values")
    if np.any(value < 0):
        raise ValueError("response must be non-negative")
    maximum = float(value.max())
    if maximum <= 0.0:
        return np.zeros_like(value, dtype=np.float32)
    return (value / maximum).astype(np.float32, copy=False)


def active_threshold_margin(response_peak: float, active_threshold: float) -> float:
    """Signed margin to the threshold actually used for this run."""

    peak = float(response_peak)
    threshold = float(active_threshold)
    if not np.isfinite(peak) or not np.isfinite(threshold):
        raise ValueError("margin inputs must be finite")
    return peak - threshold


def _stable_softmax(values: Array) -> Array:
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def tapnext_evidence(
    track_logits: Array,
    *,
    softmax_temperature: float = 0.5,
    soft_argmax_radius: int = 20,
) -> TAPNextEvidence:
    """Reproduce official TAPNext++ x/y decoding and form its joint field.

    ``track_logits`` must use the official order ``x[0:256], y[256:512]``.
    The joint field is the outer product ``p_y p_x``.  It is a deterministic
    separable lift, not a native TAPNext++ 2-D heatmap.
    """

    logits = np.asarray(track_logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size % 2 != 0 or logits.size == 0:
        raise ValueError("track_logits must be a non-empty even-length vector")
    if not np.isfinite(logits).all():
        raise ValueError("track_logits contains non-finite values")
    n = logits.size // 2
    logits_x = logits[:n]
    logits_y = logits[n:]
    index = np.arange(n, dtype=np.float64)

    def official_axis_probability(axis_logits: Array) -> Array:
        argmax = int(np.argmax(axis_logits))
        mask = np.abs(np.arange(n) - argmax) <= int(soft_argmax_radius)
        probability = _stable_softmax(axis_logits * float(softmax_temperature))
        probability *= mask
        total = float(probability.sum())
        if total <= 0.0:
            raise RuntimeError("official TAPNext++ masked probability has zero mass")
        return probability / total

    probability_x = official_axis_probability(logits_x)
    probability_y = official_axis_probability(logits_y)
    field_x = float(np.sum(probability_x * index))
    field_y = float(np.sum(probability_y * index))
    official_xy = (field_x + 0.5, field_y + 0.5)
    joint = np.outer(probability_y, probability_x).astype(np.float32)
    return TAPNextEvidence(
        joint_field=joint,
        candidate_official_xy=official_xy,
        candidate_field_xy=(field_x, field_y),
        probability_x=probability_x.astype(np.float32),
        probability_y=probability_y.astype(np.float32),
    )


def _bilinear_sample_zero(image: Array, x: Array, y: Array) -> Array:
    """Sample a 2-D image with bilinear interpolation and zero padding."""

    source = np.asarray(image, dtype=np.float32)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape")
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    output = np.zeros(x.shape, dtype=np.float64)
    height, width = source.shape

    for xx, yy, weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        output[valid] += source[yy[valid], xx[valid]] * weight[valid]
    return output.astype(np.float32)


def _global_sampling_grid(height: int, width: int, resolution: int) -> tuple[Array, Array, float, float, float]:
    scale = min(float(resolution) / float(width), float(resolution) / float(height))
    left = (float(resolution) - float(width) * scale) / 2.0
    top = (float(resolution) - float(height) * scale) / 2.0
    yy, xx = np.indices((resolution, resolution), dtype=np.float64)
    source_x = (xx + 0.5 - left) / scale - 0.5
    source_y = (yy + 0.5 - top) / scale - 0.5
    return source_x, source_y, scale, left, top


def _local_sampling_grid(
    center_xy: tuple[float, float], side: float, resolution: int
) -> tuple[Array, Array]:
    if side <= 0.0 or not np.isfinite(side):
        raise ValueError("local side must be positive and finite")
    center_x, center_y = (float(center_xy[0]), float(center_xy[1]))
    yy, xx = np.indices((resolution, resolution), dtype=np.float64)
    center_index = (int(resolution) - 1) // 2
    step = side / float(resolution)
    source_x = center_x + (xx - float(center_index)) * step
    source_y = center_y + (yy - float(center_index)) * step
    return source_x, source_y


def _splat_point(shape: tuple[int, int], x: float, y: float) -> Array:
    height, width = shape
    output = np.zeros(shape, dtype=np.float32)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    wx, wy = float(x - x0), float(y - y0)
    for xx, yy, weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x0 + 1, y0, wx * (1.0 - wy)),
        (x0, y0 + 1, (1.0 - wx) * wy),
        (x0 + 1, y0 + 1, wx * wy),
    ):
        if 0 <= xx < width and 0 <= yy < height:
            output[yy, xx] += np.float32(weight)
    return output


def _fractional_box_occupancy(
    shape: tuple[int, int], box_edges: tuple[float, float, float, float]
) -> Array:
    """Rasterize a continuous box using exact cell-intersection fractions."""

    height, width = shape
    x0, y0, x1, y1 = map(float, box_edges)
    if not all(np.isfinite((x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
        raise ValueError("box must have finite positive extent")
    output = np.zeros(shape, dtype=np.float32)
    col_start = max(0, int(np.floor(x0)))
    col_stop = min(width, int(np.ceil(x1)))
    row_start = max(0, int(np.floor(y0)))
    row_stop = min(height, int(np.ceil(y1)))
    for row in range(row_start, row_stop):
        overlap_y = max(0.0, min(float(row + 1), y1) - max(float(row), y0))
        for col in range(col_start, col_stop):
            overlap_x = max(0.0, min(float(col + 1), x1) - max(float(col), x0))
            output[row, col] = np.float32(overlap_x * overlap_y)
    return output


def _candidate_center_extent(
    candidate: tuple[float, ...], kind: CandidateKind
) -> tuple[tuple[float, float], tuple[float, float]]:
    if kind == "point":
        if len(candidate) != 2:
            raise ValueError("point candidate must be (x,y)")
        x, y = map(float, candidate)
        if not np.isfinite((x, y)).all():
            raise ValueError("point candidate must be finite")
        return (x, y), (0.0, 0.0)
    if kind == "box":
        if len(candidate) != 4:
            raise ValueError("box candidate must be edge coordinates (x0,y0,x1,y1)")
        x0, y0, x1, y1 = map(float, candidate)
        if not all(np.isfinite((x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
            raise ValueError("box candidate must have finite positive extent")
        # Box inputs use edge coordinates.  Convert their center to response
        # cell-center coordinates before sampling the response field.
        return ((x0 + x1) / 2.0 - 0.5, (y0 + y1) / 2.0 - 0.5), (x1 - x0, y1 - y0)
    raise ValueError(f"unknown candidate kind: {kind}")


def canonicalize_crep(
    response: Array,
    candidate: tuple[float, ...],
    *,
    candidate_kind: CandidateKind,
    native_margin: float,
    resolution: int,
    rho: float,
) -> CREPPacket:
    """Create deterministic global and local CREP spatial views."""

    if int(resolution) != resolution or int(resolution) <= 1:
        raise ValueError("resolution must be an integer greater than one")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must lie in (0,1]")
    if not np.isfinite(native_margin):
        raise ValueError("native_margin must be finite")

    normalized = peak_normalize(response)
    height, width = normalized.shape
    center, extent = _candidate_center_extent(candidate, candidate_kind)
    ones = np.ones_like(normalized, dtype=np.float32)

    gx, gy, scale, left, top = _global_sampling_grid(height, width, int(resolution))
    global_response = _bilinear_sample_zero(normalized, gx, gy)
    global_valid = _bilinear_sample_zero(ones, gx, gy)

    if candidate_kind == "point":
        point_x = left + scale * (center[0] + 0.5) - 0.5
        point_y = top + scale * (center[1] + 0.5) - 0.5
        global_candidate = _splat_point((int(resolution), int(resolution)), point_x, point_y)
    else:
        x0, y0, x1, y1 = map(float, candidate)
        global_box = (
            left + scale * x0,
            top + scale * y0,
            left + scale * x1,
            top + scale * y1,
        )
        global_candidate = _fractional_box_occupancy(
            (int(resolution), int(resolution)), global_box
        )

    # The candidate is sampled exactly at a designated center cell.  For a box,
    # one-cell slack keeps the full fractional occupancy inside an even-sized
    # canonical grid rather than clipping half a boundary cell.
    box_scale = float(resolution) / float(resolution - 1)
    local_side = max(
        float(rho) * min(height, width),
        extent[0] * box_scale,
        extent[1] * box_scale,
    )
    if local_side <= 0.0:
        local_side = float(rho) * min(height, width)
    lx, ly = _local_sampling_grid(center, local_side, int(resolution))
    local_response = _bilinear_sample_zero(normalized, lx, ly)
    local_valid = _bilinear_sample_zero(ones, lx, ly)

    if candidate_kind == "point":
        local_center = (float((int(resolution) - 1) // 2),) * 2
        local_candidate = _splat_point(
            (int(resolution), int(resolution)), local_center[0], local_center[1]
        )
    else:
        x0, y0, x1, y1 = map(float, candidate)
        center_edge_x = center[0] + 0.5
        center_edge_y = center[1] + 0.5
        center_edge_canonical = float((int(resolution) - 1) // 2) + 0.5
        step = local_side / float(resolution)
        local_box = (
            center_edge_canonical + (x0 - center_edge_x) / step,
            center_edge_canonical + (y0 - center_edge_y) / step,
            center_edge_canonical + (x1 - center_edge_x) / step,
            center_edge_canonical + (y1 - center_edge_y) / step,
        )
        local_candidate = _fractional_box_occupancy(
            (int(resolution), int(resolution)), local_box
        )

    return CREPPacket(
        global_view=SpatialView(global_response, global_candidate, global_valid),
        local_view=SpatialView(local_response, local_candidate, local_valid),
        native_margin=float(native_margin),
        candidate_kind=candidate_kind,
        resolution=int(resolution),
        rho=float(rho),
    )


def packet_vector(packet: CREPPacket, contract: Literal["M", "G", "L", "LG", "LG-M"]) -> Array:
    """Flatten one authorized Phase-3.5 diagnostic contract."""

    global_channels = (
        packet.global_view.response,
        packet.global_view.candidate_support,
        packet.global_view.valid_support,
    )
    local_channels = (
        packet.local_view.response,
        packet.local_view.candidate_support,
        packet.local_view.valid_support,
    )
    if contract == "M":
        arrays: tuple[Array, ...] = ()
        include_margin = True
    elif contract == "G":
        arrays = global_channels
        include_margin = True
    elif contract == "L":
        arrays = local_channels
        include_margin = True
    elif contract == "LG":
        arrays = global_channels + local_channels
        include_margin = True
    elif contract == "LG-M":
        arrays = global_channels + local_channels
        include_margin = False
    else:
        raise ValueError(f"unsupported CREP contract: {contract}")
    pieces = [array.astype(np.float32, copy=False).ravel() for array in arrays]
    if include_margin:
        pieces.append(np.asarray([packet.native_margin], dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
