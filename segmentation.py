"""Cell segmentation module for FluoQuant.

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

SEGMENTATION_EXPORT_OPTIONS = (
    ("cells_csv", "单细胞 CSV"),
    ("labels", "标签掩膜 TIF"),
    ("segmentation_overlay", "边界 Overlay PNG"),
    ("segmentation_metadata", "参数/汇总 JSON"),
)

METRIC_OPTIONS = (
    ("morph_basic", "基本形态（面积、等效直径）"),
    ("morph_shape", "形态细节（周长、圆度、长宽比、实度）"),
    ("intensity_extra", "强度统计（median / max / integrated）"),
    ("compartments", "核/质或核/核周区室指标"),
)


@dataclass
class SegmentationResult:
    labels: np.ndarray  # int32 label mask of the segmented region (cells)
    offset: tuple[int, int]  # (x, y) of the region inside the full image
    source_key: str
    source_label: str
    method: str  # "classical" | "cellpose"
    params: dict
    nucleus_labels: np.ndarray | None = field(default=None, repr=False)
    _boundaries: np.ndarray | None = field(default=None, repr=False)
    _nucleus_boundaries: np.ndarray | None = field(default=None, repr=False)

    @property
    def count(self) -> int:
        return int(self.labels.max())

    @property
    def dual(self) -> bool:
        return self.nucleus_labels is not None

    def boundary_mask(self) -> np.ndarray:
        if self._boundaries is None:
            from skimage.segmentation import find_boundaries

            self._boundaries = find_boundaries(self.labels, mode="outer")
        return self._boundaries

    def nucleus_boundary_mask(self) -> np.ndarray | None:
        if self.nucleus_labels is None:
            return None
        if self._nucleus_boundaries is None:
            from skimage.segmentation import find_boundaries

            self._nucleus_boundaries = find_boundaries(self.nucleus_labels, mode="outer")
        return self._nucleus_boundaries

    def _place(self, region: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        x0, y0 = self.offset
        mask[y0 : y0 + region.shape[0], x0 : x0 + region.shape[1]] = region
        return mask

    def full_boundary_mask(self, shape: tuple[int, int]) -> np.ndarray:
        return self._place(self.boundary_mask(), shape)

    def full_nucleus_boundary_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        region = self.nucleus_boundary_mask()
        return self._place(region, shape) if region is not None else None


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


def segment_classical_dual(
    nucleus: np.ndarray,
    cell_channel: np.ndarray,
    pixel_size_um: float,
    min_area_um2: float = 20.0,
    smooth_um: float = 0.5,
    cell_diameter_um: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nucleus + whole-cell segmentation with classical methods.

    Nuclei come from Otsu + watershed on the nuclear channel; cells are
    grown from each nucleus with a seeded watershed on the cell-boundary
    channel (e.g. F-actin), limited to cell_diameter_um/2 beyond the
    nucleus. Returns (cell_labels, nucleus_labels).
    """
    from skimage.filters import gaussian
    from skimage.segmentation import watershed

    nuclei = segment_classical(
        nucleus, pixel_size_um, min_area_um2=min_area_um2, smooth_um=smooth_um
    )
    if nuclei.max() == 0:
        return nuclei.copy(), nuclei
    sigma_px = max(0.5, smooth_um / pixel_size_um)
    elevation = gaussian(
        np.asarray(cell_channel, dtype=np.float32), sigma=sigma_px, preserve_range=True
    )
    max_px = max(2, int(round(cell_diameter_um / 2.0 / pixel_size_um)))
    distance = ndimage.distance_transform_edt(nuclei == 0)
    cells = watershed(elevation, nuclei, mask=distance <= max_px).astype(np.int32)
    return cells, nuclei


def segment_cellpose_dual(
    nucleus: np.ndarray,
    cell_channel: np.ndarray,
    pixel_size_um: float,
    diameter_um: float = 20.0,
    flow_threshold: float = 0.4,
    min_area_um2: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nucleus + whole-cell segmentation with Cellpose.

    Whole cells are segmented from a two-channel stack
    [cell boundary/cytoplasm, nucleus]; nuclei from the nuclear channel
    alone (diameter assumed to be half the cell diameter).
    Returns (cell_labels, nucleus_labels).
    """
    try:
        from cellpose import models
    except ImportError as error:
        raise ValueError("Cellpose 未安装：python -m pip install cellpose") from error
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is None:
        try:
            _CELLPOSE_MODEL = models.CellposeModel(gpu=True)
        except Exception:
            _CELLPOSE_MODEL = models.CellposeModel(gpu=False)
    min_px = max(4, int(round(min_area_um2 / (pixel_size_um**2))))
    diameter_px = diameter_um / pixel_size_um if diameter_um > 0 else None
    stack = np.stack(
        [
            np.asarray(cell_channel, dtype=np.float32),
            np.asarray(nucleus, dtype=np.float32),
        ],
        axis=-1,
    )
    cell_masks, _flows, _styles = _CELLPOSE_MODEL.eval(
        stack,
        channel_axis=-1,
        diameter=diameter_px,
        flow_threshold=flow_threshold,
        min_size=min_px,
    )
    nucleus_masks, _flows, _styles = _CELLPOSE_MODEL.eval(
        np.asarray(nucleus, dtype=np.float32),
        diameter=diameter_px / 2.0 if diameter_px else None,
        flow_threshold=flow_threshold,
        min_size=min_px,
    )
    return np.asarray(cell_masks, dtype=np.int32), np.asarray(nucleus_masks, dtype=np.int32)


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
    ratio_names: set[str] | None = None,
    metrics: set[str] | None = None,
) -> tuple[list[str], list[list[float]]]:
    """Per-cell morphology, intensity, and nuclear/cytoplasm statistics.

    channels: (column name, full-image intensity array) pairs.
    ring_um: width of the perinuclear ring grown outward from each
        segmented object (usually a nucleus); 0 disables ring columns.
        The ring approximates cytoplasm, so the ratio is reported as
        nuc_peri_ratio (nucleus / perinuclear), not a true N/C ratio.
    ratio_names: channel names that get perinuclear/ratio columns;
        None means every channel, an empty set means none.
    metrics: METRIC_OPTIONS keys to include (None means all); cell id,
        centroid, and per-channel mean are always present.
    Returns (header, rows) ready for CSV writing.
    """
    from skimage.measure import regionprops
    from skimage.segmentation import expand_labels

    include = lambda key: metrics is None or key in metrics  # noqa: E731
    labels = result.labels
    x0, y0 = result.offset
    height, width = labels.shape
    header = ["cell_id", "centroid_x_um", "centroid_y_um"]
    if include("morph_basic"):
        header += ["area_um2", "equivalent_diameter_um"]
    if include("morph_shape"):
        header += ["perimeter_um", "circularity", "aspect_ratio", "solidity"]
    rows: list[list[float]] = []
    for prop in regionprops(labels):
        row = [
            float(prop.label),
            (x0 + prop.centroid[1]) * pixel_size_um,
            (y0 + prop.centroid[0]) * pixel_size_um,
        ]
        if include("morph_basic"):
            row += [
                prop.area * pixel_size_um**2,
                prop.equivalent_diameter * pixel_size_um,
            ]
        if include("morph_shape"):
            perimeter_px = float(prop.perimeter)
            circularity = (
                min(1.0, 4.0 * np.pi * prop.area / perimeter_px**2)
                if perimeter_px > 0
                else float("nan")
            )
            minor = float(prop.minor_axis_length)
            aspect = (
                float(prop.major_axis_length) / minor if minor > 0 else float("nan")
            )
            row += [perimeter_px * pixel_size_um, circularity, aspect, float(prop.solidity)]
        rows.append(row)
    index = np.arange(1, labels.max() + 1)
    wants_ratio = lambda name: ratio_names is None or name in ratio_names  # noqa: E731
    ring = None
    if (
        include("compartments")
        and ring_um > 0
        and labels.max() > 0
        and any(wants_ratio(name) for name, _data in channels)
    ):
        ring_px = max(1, int(round(ring_um / pixel_size_um)))
        expanded = expand_labels(labels, distance=ring_px)
        ring = np.where(labels > 0, 0, expanded)
        ring_sizes = ndimage.sum(np.ones_like(ring), ring, index)
    for name, data in channels:
        crop = np.asarray(data[y0 : y0 + height, x0 : x0 + width], dtype=np.float32)
        means = ndimage.mean(crop, labels, index)
        header.append(f"{name}_mean")
        extra = include("intensity_extra")
        if extra:
            medians = ndimage.median(crop, labels, index)
            maxima = ndimage.maximum(crop, labels, index)
            sums = ndimage.sum(crop, labels, index)
            header += [f"{name}_median", f"{name}_max", f"{name}_integrated"]
        channel_ring = ring if wants_ratio(name) else None
        if channel_ring is not None:
            header += [f"{name}_perinuclear_mean", f"{name}_nuc_peri_ratio"]
            with np.errstate(invalid="ignore", divide="ignore"):
                ring_means = np.where(
                    ring_sizes > 0, ndimage.mean(crop, channel_ring, index), np.nan
                )
        for position, (row, mean) in enumerate(zip(rows, means)):
            row.append(float(mean))
            if extra:
                row += [
                    float(medians[position]),
                    float(maxima[position]),
                    float(sums[position]),
                ]
            if channel_ring is not None:
                cyto = float(ring_means[position])
                ratio = float(mean) / cyto if np.isfinite(cyto) and cyto > 0 else float("nan")
                row += [cyto, ratio]
    return header, rows


def measure_cells_dual(
    result: SegmentationResult,
    channels: list[tuple[str, np.ndarray]],
    pixel_size_um: float,
    ratio_names: set[str] | None = None,
    metrics: set[str] | None = None,
) -> tuple[list[str], list[list[float]]]:
    """Per-cell statistics with true nucleus/cytoplasm compartments.

    The nucleus compartment of cell i is every nucleus pixel lying inside
    cell i; the cytoplasm is the rest of the cell. metrics selects
    METRIC_OPTIONS groups (None means all). Returns (header, rows).
    """
    from skimage.measure import regionprops

    assert result.nucleus_labels is not None
    include = lambda key: metrics is None or key in metrics  # noqa: E731
    cell_labels = result.labels
    x0, y0 = result.offset
    height, width = cell_labels.shape
    nucleus_region = np.where(result.nucleus_labels > 0, cell_labels, 0)
    cyto_region = np.where(result.nucleus_labels > 0, 0, cell_labels)
    index = np.arange(1, cell_labels.max() + 1)
    ones = np.ones(cell_labels.shape, dtype=np.float32)
    nucleus_sizes = ndimage.sum(ones, nucleus_region, index)
    cyto_sizes = ndimage.sum(ones, cyto_region, index)
    wants_ratio = lambda name: ratio_names is None or name in ratio_names  # noqa: E731

    header = ["cell_id", "centroid_x_um", "centroid_y_um"]
    if include("morph_basic"):
        header += ["cell_area_um2", "equivalent_diameter_um"]
    if include("compartments"):
        header += ["nucleus_area_um2", "cyto_area_um2", "nc_area_ratio"]
    if include("morph_shape"):
        header += ["perimeter_um", "circularity", "aspect_ratio", "solidity"]
    rows: list[list[float]] = []
    for position, prop in enumerate(regionprops(cell_labels)):
        row = [
            float(prop.label),
            (x0 + prop.centroid[1]) * pixel_size_um,
            (y0 + prop.centroid[0]) * pixel_size_um,
        ]
        cell_area = prop.area * pixel_size_um**2
        if include("morph_basic"):
            row += [cell_area, prop.equivalent_diameter * pixel_size_um]
        if include("compartments"):
            nucleus_area = float(nucleus_sizes[position]) * pixel_size_um**2
            row += [
                nucleus_area,
                float(cyto_sizes[position]) * pixel_size_um**2,
                nucleus_area / cell_area if cell_area > 0 else float("nan"),
            ]
        if include("morph_shape"):
            perimeter_px = float(prop.perimeter)
            circularity = (
                min(1.0, 4.0 * np.pi * prop.area / perimeter_px**2)
                if perimeter_px > 0
                else float("nan")
            )
            minor = float(prop.minor_axis_length)
            aspect = (
                float(prop.major_axis_length) / minor if minor > 0 else float("nan")
            )
            row += [perimeter_px * pixel_size_um, circularity, aspect, float(prop.solidity)]
        rows.append(row)
    for name, data in channels:
        crop = np.asarray(data[y0 : y0 + height, x0 : x0 + width], dtype=np.float32)
        cell_means = ndimage.mean(crop, cell_labels, index)
        header.append(f"{name}_cell_mean")
        extra = include("intensity_extra")
        if extra:
            cell_medians = ndimage.median(crop, cell_labels, index)
            cell_maxima = ndimage.maximum(crop, cell_labels, index)
            cell_sums = ndimage.sum(crop, cell_labels, index)
            header += [
                f"{name}_cell_median",
                f"{name}_cell_max",
                f"{name}_cell_integrated",
            ]
        compartments = include("compartments") and wants_ratio(name)
        if compartments:
            header += [f"{name}_nuc_mean", f"{name}_cyto_mean", f"{name}_nuc_cyto_ratio"]
            with np.errstate(invalid="ignore", divide="ignore"):
                nucleus_means = np.where(
                    nucleus_sizes > 0,
                    ndimage.mean(crop, nucleus_region, index),
                    np.nan,
                )
                cyto_means = np.where(
                    cyto_sizes > 0, ndimage.mean(crop, cyto_region, index), np.nan
                )
        for position, (row, mean) in enumerate(zip(rows, cell_means)):
            row.append(float(mean))
            if extra:
                row += [
                    float(cell_medians[position]),
                    float(cell_maxima[position]),
                    float(cell_sums[position]),
                ]
            if compartments:
                nuc_mean = float(nucleus_means[position])
                cyto_mean = float(cyto_means[position])
                ratio = (
                    nuc_mean / cyto_mean
                    if np.isfinite(nuc_mean) and np.isfinite(cyto_mean) and cyto_mean > 0
                    else float("nan")
                )
                row += [nuc_mean, cyto_mean, ratio]
    return header, rows


def make_segmentation_overlay(
    composite_rgb: np.ndarray,
    result: SegmentationResult,
    color: tuple[int, int, int] = (255, 214, 0),
    nucleus_color: tuple[int, int, int] = (0, 229, 255),
) -> np.ndarray:
    overlay = np.array(composite_rgb, copy=True)
    overlay[result.full_boundary_mask(overlay.shape[:2])] = color
    nucleus_mask = result.full_nucleus_boundary_mask(overlay.shape[:2])
    if nucleus_mask is not None:
        overlay[nucleus_mask] = nucleus_color
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
    selected: set[str] | None = None,
    ratio_names: set[str] | None = None,
    metrics: set[str] | None = None,
) -> dict[str, Path]:
    import tifffile
    from PIL import Image

    if selected is None:
        selected = {key for key, _label in SEGMENTATION_EXPORT_OPTIONS}
    if not selected:
        raise ValueError("请至少勾选一项要导出的内容。")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_cells.csv"
    overlay_path = output_dir / f"{stem}_segmentation_overlay.png"
    labels_path = output_dir / f"{stem}_labels.tif"
    metadata_path = output_dir / f"{stem}_segmentation.json"

    exported: dict[str, Path] = {}
    if result.dual:
        header, rows = measure_cells_dual(
            result, channels, pixel_size_um, ratio_names=ratio_names, metrics=metrics
        )
    else:
        header, rows = measure_cells(
            result,
            channels,
            pixel_size_um,
            ring_um=ring_um,
            ratio_names=ratio_names,
            metrics=metrics,
        )
    if "cells_csv" in selected:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        exported["cells_csv"] = csv_path

    if "segmentation_overlay" in selected:
        overlay = make_segmentation_overlay(composite_rgb, result)
        Image.fromarray(overlay).save(overlay_path, dpi=(300, 300))
        exported["segmentation_overlay"] = overlay_path
    if "labels" in selected:
        label_dtype = np.uint16 if result.count < 65536 else np.int32
        tifffile.imwrite(labels_path, result.labels.astype(label_dtype))
        exported["labels"] = labels_path
        if result.nucleus_labels is not None:
            nuclei_path = output_dir / f"{stem}_labels_nuclei.tif"
            tifffile.imwrite(nuclei_path, result.nucleus_labels.astype(label_dtype))
            exported["labels_nuclei"] = nuclei_path

    region_area_mm2 = (
        result.labels.shape[0] * result.labels.shape[1] * (pixel_size_um / 1000.0) ** 2
    )
    summary: dict[str, object] = {
        "cell_count": result.count,
        "cell_density_per_mm2": result.count / region_area_mm2 if region_area_mm2 > 0 else None,
    }
    if rows:
        table = np.asarray(rows, dtype=float)
        summary_columns = (
            ("cell_area_um2", "nucleus_area_um2", "nc_area_ratio", "equivalent_diameter_um", "circularity")
            if result.dual
            else ("area_um2", "equivalent_diameter_um", "circularity")
        )
        for column_name in summary_columns:
            if column_name not in header:
                continue
            column = table[:, header.index(column_name)]
            column = column[np.isfinite(column)]
            if column.size:
                summary[f"mean_{column_name}"] = float(np.mean(column))
        for name, _data in channels:
            for key in (
                f"{name}_mean",
                f"{name}_cell_mean",
                f"{name}_nuc_cyto_ratio",
                f"{name}_nuc_peri_ratio",
            ):
                if key in header:
                    values = table[:, header.index(key)]
                    values = values[np.isfinite(values)]
                    if values.size:
                        summary[f"mean_{key}"] = float(np.mean(values))
    metadata = {
        "source_image": str(source_image),
        "pixel_size_um": pixel_size_um,
        "segmentation_channel": result.source_label,
        "method": result.method,
        "mode": "nucleus+cell" if result.dual else "single",
        "nucleus_count": int(result.nucleus_labels.max()) if result.dual else None,
        "params": result.params,
        "cyto_ring_um": ring_um,
        "metric_groups": sorted(metrics) if metrics is not None else "all",
        "region_offset_xy_px": list(result.offset),
        "region_size_xy_px": [result.labels.shape[1], result.labels.shape[0]],
        "cell_count": result.count,
        "measured_channels": [name for name, _data in channels],
        "summary": summary,
        "note": (
            "labels.tif is the label mask (0 = background); cell_id in the CSV matches label values. "
            "Single mode: perinuclear columns sample a ring grown outward from each segmented object; "
            "nuc_peri_ratio = object mean / ring mean (cytoplasm proxy, not a true N/C ratio). "
            "Dual mode: cyto columns are the true cytoplasm compartment (cell minus nucleus) and "
            "nuc_cyto_ratio = nucleus mean / cytoplasm mean."
        ),
    }
    if "segmentation_metadata" in selected:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        exported["segmentation_metadata"] = metadata_path
    return exported
