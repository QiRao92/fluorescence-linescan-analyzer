"""Cell segmentation module for the Fluorescence Line-scan Analyzer.

Pure computation: no GUI imports. Provides a classical
(Otsu + watershed) pipeline with no extra dependencies and an optional
Cellpose (deep learning) pipeline, plus per-cell intensity measurement
and export helpers.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

# Anaconda ships multiple OpenMP runtimes (MKL + torch); without this
# flag importing torch/cellpose next to numpy aborts the process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_CELLPOSE_MODEL = None


@dataclass
class SegmentationResult:
    labels: np.ndarray  # int32 label mask of the segmented region
    offset: tuple[int, int]  # (x, y) of the region inside the full image
    source_key: str
    source_label: str
    method: str  # "classical" | "cellpose"
    params: dict
    _boundaries: np.ndarray | None = field(default=None, repr=False)

    @property
    def count(self) -> int:
        return int(self.labels.max())

    def boundary_mask(self) -> np.ndarray:
        if self._boundaries is None:
            from skimage.segmentation import find_boundaries

            self._boundaries = find_boundaries(self.labels, mode="outer")
        return self._boundaries

    def full_boundary_mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        x0, y0 = self.offset
        region = self.boundary_mask()
        mask[y0 : y0 + region.shape[0], x0 : x0 + region.shape[1]] = region
        return mask


def segment_classical(
    intensity: np.ndarray,
    pixel_size_um: float,
    min_area_um2: float = 20.0,
    smooth_um: float = 0.5,
) -> np.ndarray:
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    values = np.asarray(intensity, dtype=np.float32)
    sigma_px = max(0.5, smooth_um / pixel_size_um)
    smoothed = gaussian(values, sigma=sigma_px, preserve_range=True)
    finite = smoothed[np.isfinite(smoothed)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return np.zeros(values.shape, dtype=np.int32)
    mask = smoothed > threshold_otsu(smoothed)
    mask = ndimage.binary_fill_holes(mask)
    min_px = max(4, int(round(min_area_um2 / (pixel_size_um**2))))
    mask = remove_small_objects(mask, min_size=min_px)
    if not mask.any():
        return np.zeros(values.shape, dtype=np.int32)
    distance = ndimage.distance_transform_edt(mask)
    min_distance = max(3, int(round(np.sqrt(min_px / np.pi))))
    coordinates = peak_local_max(distance, min_distance=min_distance, labels=mask)
    if len(coordinates) == 0:
        labels, _count = ndimage.label(mask)
        return labels.astype(np.int32)
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    labels = watershed(-distance, markers, mask=mask).astype(np.int32)
    return _drop_small_labels(labels, min_px)


def segment_cellpose(
    intensity: np.ndarray,
    pixel_size_um: float,
    diameter_um: float = 0.0,
    flow_threshold: float = 0.4,
    min_area_um2: float = 20.0,
) -> np.ndarray:
    try:
        from cellpose import models
    except ImportError as error:
        raise ValueError(
            "Cellpose 未安装：python -m pip install cellpose"
        ) from error
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is None:
        try:
            _CELLPOSE_MODEL = models.CellposeModel(gpu=True)
        except Exception:
            _CELLPOSE_MODEL = models.CellposeModel(gpu=False)
    min_px = max(4, int(round(min_area_um2 / (pixel_size_um**2))))
    diameter_px = diameter_um / pixel_size_um if diameter_um > 0 else None
    masks, _flows, _styles = _CELLPOSE_MODEL.eval(
        np.asarray(intensity, dtype=np.float32),
        diameter=diameter_px,
        flow_threshold=flow_threshold,
        min_size=min_px,
    )
    return np.asarray(masks, dtype=np.int32)


def _drop_small_labels(labels: np.ndarray, min_px: int) -> np.ndarray:
    if labels.max() == 0:
        return labels
    counts = np.bincount(labels.ravel())
    keep = counts >= min_px
    keep[0] = False
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1)
    return mapping[labels]


def measure_cells(
    result: SegmentationResult,
    channels: list[tuple[str, np.ndarray]],
    pixel_size_um: float,
    ring_um: float = 1.5,
) -> tuple[list[str], list[list[float]]]:
    """Per-cell morphology, intensity, and nuclear/cytoplasm statistics.

    channels: (column name, full-image intensity array) pairs.
    ring_um: width of the cytoplasmic ring grown outward from each
        segmented object (usually a nucleus); 0 disables ring columns.
    Returns (header, rows) ready for CSV writing.
    """
    from skimage.measure import regionprops
    from skimage.segmentation import expand_labels

    labels = result.labels
    x0, y0 = result.offset
    height, width = labels.shape
    header = [
        "cell_id",
        "centroid_x_um",
        "centroid_y_um",
        "area_um2",
        "equivalent_diameter_um",
        "perimeter_um",
        "circularity",
        "aspect_ratio",
        "solidity",
    ]
    rows: list[list[float]] = []
    for prop in regionprops(labels):
        perimeter_px = float(prop.perimeter)
        circularity = (
            min(1.0, 4.0 * np.pi * prop.area / perimeter_px**2)
            if perimeter_px > 0
            else float("nan")
        )
        minor = float(prop.minor_axis_length)
        aspect = float(prop.major_axis_length) / minor if minor > 0 else float("nan")
        rows.append(
            [
                float(prop.label),
                (x0 + prop.centroid[1]) * pixel_size_um,
                (y0 + prop.centroid[0]) * pixel_size_um,
                prop.area * pixel_size_um**2,
                prop.equivalent_diameter * pixel_size_um,
                perimeter_px * pixel_size_um,
                circularity,
                aspect,
                float(prop.solidity),
            ]
        )
    index = np.arange(1, labels.max() + 1)
    ring = None
    if ring_um > 0 and labels.max() > 0:
        ring_px = max(1, int(round(ring_um / pixel_size_um)))
        expanded = expand_labels(labels, distance=ring_px)
        ring = np.where(labels > 0, 0, expanded)
        ring_sizes = ndimage.sum(np.ones_like(ring), ring, index)
    for name, data in channels:
        crop = np.asarray(data[y0 : y0 + height, x0 : x0 + width], dtype=np.float32)
        means = ndimage.mean(crop, labels, index)
        medians = ndimage.median(crop, labels, index)
        maxima = ndimage.maximum(crop, labels, index)
        sums = ndimage.sum(crop, labels, index)
        header += [f"{name}_mean", f"{name}_median", f"{name}_max", f"{name}_integrated"]
        if ring is not None:
            header += [f"{name}_cyto_mean", f"{name}_nuc_cyto_ratio"]
            with np.errstate(invalid="ignore", divide="ignore"):
                ring_means = np.where(
                    ring_sizes > 0, ndimage.mean(crop, ring, index), np.nan
                )
        for position, (row, mean, median, maximum, total) in enumerate(
            zip(rows, means, medians, maxima, sums)
        ):
            row += [float(mean), float(median), float(maximum), float(total)]
            if ring is not None:
                cyto = float(ring_means[position])
                ratio = float(mean) / cyto if np.isfinite(cyto) and cyto > 0 else float("nan")
                row += [cyto, ratio]
    return header, rows


def make_segmentation_overlay(
    composite_rgb: np.ndarray,
    result: SegmentationResult,
    color: tuple[int, int, int] = (255, 214, 0),
) -> np.ndarray:
    overlay = np.array(composite_rgb, copy=True)
    mask = result.full_boundary_mask(overlay.shape[:2])
    overlay[mask] = color
    return overlay


def export_segmentation(
    result: SegmentationResult,
    composite_rgb: np.ndarray,
    channels: list[tuple[str, np.ndarray]],
    pixel_size_um: float,
    source_image: Path,
    output_dir: Path,
    stem: str,
    ring_um: float = 1.5,
) -> dict[str, Path]:
    import tifffile
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_cells.csv"
    overlay_path = output_dir / f"{stem}_segmentation_overlay.png"
    labels_path = output_dir / f"{stem}_labels.tif"
    metadata_path = output_dir / f"{stem}_segmentation.json"

    header, rows = measure_cells(result, channels, pixel_size_um, ring_um=ring_um)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    overlay = make_segmentation_overlay(composite_rgb, result)
    Image.fromarray(overlay).save(overlay_path, dpi=(300, 300))
    tifffile.imwrite(labels_path, result.labels.astype(np.uint16 if result.count < 65536 else np.int32))

    region_area_mm2 = (
        result.labels.shape[0] * result.labels.shape[1] * (pixel_size_um / 1000.0) ** 2
    )
    summary: dict[str, object] = {
        "cell_count": result.count,
        "cell_density_per_mm2": result.count / region_area_mm2 if region_area_mm2 > 0 else None,
    }
    if rows:
        table = np.asarray(rows, dtype=float)
        for column_name in ("area_um2", "equivalent_diameter_um", "circularity"):
            column = table[:, header.index(column_name)]
            summary[f"mean_{column_name}"] = float(np.nanmean(column))
        for name, _data in channels:
            key = f"{name}_mean"
            if key in header:
                summary[f"mean_{key}"] = float(np.nanmean(table[:, header.index(key)]))
            ratio_key = f"{name}_nuc_cyto_ratio"
            if ratio_key in header:
                summary[f"mean_{ratio_key}"] = float(np.nanmean(table[:, header.index(ratio_key)]))
    metadata = {
        "source_image": str(source_image),
        "pixel_size_um": pixel_size_um,
        "segmentation_channel": result.source_label,
        "method": result.method,
        "params": result.params,
        "cyto_ring_um": ring_um,
        "region_offset_xy_px": list(result.offset),
        "region_size_xy_px": [result.labels.shape[1], result.labels.shape[0]],
        "cell_count": result.count,
        "measured_channels": [name for name, _data in channels],
        "summary": summary,
        "note": (
            "labels.tif is the label mask (0 = background); cell_id in the CSV matches label values. "
            "cyto columns sample a ring grown outward from each segmented object (usually a nucleus); "
            "nuc_cyto_ratio = object mean / ring mean."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "cells_csv": csv_path,
        "segmentation_overlay": overlay_path,
        "labels": labels_path,
        "segmentation_metadata": metadata_path,
    }
