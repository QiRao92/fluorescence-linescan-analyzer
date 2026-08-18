from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_API", "PySide6")

import numpy as np
import tifffile
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter1d
from skimage.measure import profile_line


APP_TITLE = "Fluorescence Line-scan Analyzer"
HNF4A_COLOR = "#D55E00"
DAPI_COLOR = "#0072B2"
ROI_COLOR = "#FFD400"
LINE_COLOR = "#FFFFFF"
SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}


def configure_fonts(app: QApplication) -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    qt_family = "Arial"
    for font_path in candidates:
        if not font_path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        if families:
            qt_family = families[0]
            break
    app.setFont(QFont(qt_family, 9))
    for font_path in candidates:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            plot_family = font_manager.FontProperties(fname=str(font_path)).get_name()
            rcParams["font.sans-serif"] = [plot_family, "Arial", "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            break
        except RuntimeError:
            continue


def configure_light_theme(app: QApplication) -> None:
    """Use deterministic light colors instead of inheriting Windows dark mode."""
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#F3F4F6",
        QPalette.ColorRole.WindowText: "#111827",
        QPalette.ColorRole.Base: "#FFFFFF",
        QPalette.ColorRole.AlternateBase: "#F8FAFC",
        QPalette.ColorRole.ToolTipBase: "#FFFFFF",
        QPalette.ColorRole.ToolTipText: "#111827",
        QPalette.ColorRole.Text: "#111827",
        QPalette.ColorRole.Button: "#FFFFFF",
        QPalette.ColorRole.ButtonText: "#111827",
        QPalette.ColorRole.BrightText: "#B91C1C",
        QPalette.ColorRole.Link: "#0F766E",
        QPalette.ColorRole.Highlight: "#0F766E",
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.PlaceholderText: "#6B7280",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#9CA3AF")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor("#9CA3AF"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#F3F4F6")
    )
    app.setPalette(palette)


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "image"


def safe_label(text: str) -> str:
    return re.sub(r"[^\w-]+", "_", text, flags=re.UNICODE).strip("_") or "signal"


def display_name(path: Path) -> str:
    name = path.name
    name = name.replace("Experiment-26-", "")
    name = name.replace("-Airyscan Processing-", " / AS-")
    name = name.replace("_s1c1-3.tif", "")
    name = name.replace("_c1-3.tif", "")
    return name


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values
    values = values.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    if float(np.min(finite)) >= 0 and float(np.max(finite)) <= 255:
        return np.round(np.clip(values, 0, 255)).astype(np.uint8)
    low, high = np.percentile(finite, (0.2, 99.8))
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def read_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        array = np.asarray(Image.open(path))
    array = np.asarray(array)
    while array.ndim > 3:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    return array


def as_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        gray = normalize_uint8(array)
        return np.repeat(gray[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"不支持的图像维度：{array.shape}")
    if array.shape[-1] == 1:
        gray = normalize_uint8(array[..., 0])
        return np.repeat(gray[..., None], 3, axis=-1)
    return normalize_uint8(array[..., :3])


def as_intensity(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        return np.max(np.asarray(array[..., :3], dtype=np.float32), axis=-1)
    raise ValueError(f"不支持的通道图像维度：{array.shape}")


def resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return np.asarray(array, dtype=np.float32)
    source = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    resized = source.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def sibling_channel_paths(composite: Path) -> tuple[Path, Path, Path] | None:
    name = composite.name
    if name.endswith("_s1c1-3.tif"):
        prefix = name[: -len("_s1c1-3.tif")]
        paths = tuple(composite.with_name(f"{prefix}_s1c{i}.tif") for i in (1, 2, 3))
    elif name.endswith("_c1-3.tif"):
        prefix = name[: -len("_c1-3.tif")]
        paths = tuple(composite.with_name(f"{prefix}_c{i}.tif") for i in (1, 2, 3))
    else:
        return None
    if not all(path.exists() for path in paths):
        return None
    return paths  # type: ignore[return-value]


def parse_pixel_size(path: Path, image_width: int) -> tuple[float, str]:
    for metadata_path in sorted(path.parent.glob("*.tif_metadata.xml")):
        try:
            root = ET.parse(metadata_path).getroot()
            for distance in root.findall(".//Scaling/Items/Distance"):
                if distance.attrib.get("Id") != "X":
                    continue
                value = distance.find("Value")
                if value is not None and value.text:
                    return float(value.text) * 1e6, metadata_path.name
        except (ET.ParseError, OSError, ValueError):
            continue
    if image_width == 4000:
        return 0.06175506268081, "按匹配的 4000 px ZEISS 视野推断"
    if image_width == 2000:
        return 0.12354834012813039, "按匹配的 2000 px ZEISS 视野推断"
    return 1.0, "未找到标定；请手动输入"


@dataclass
class ImageRecord:
    path: Path
    name: str
    signal1_name: str = "HNF4A"
    signal2_name: str = "DAPI"
    pixel_size_um: float = 1.0
    pixel_source: str = "尚未读取"
    rgb: np.ndarray | None = None
    factin: np.ndarray | None = None
    hnf4a: np.ndarray | None = None
    dapi: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None
    line: tuple[float, float, float, float] | None = None
    distance_um: np.ndarray | None = None
    hnf4a_profile: np.ndarray | None = None
    dapi_profile: np.ndarray | None = None
    line_width_px: int | None = None
    analysis_line_width_um: float | None = None
    analysis_smoothing_sigma: float | None = None
    analysis_background_percentile: float | None = None
    dirty: bool = True

    @property
    def loaded(self) -> bool:
        return self.rgb is not None

    @property
    def analyzed(self) -> bool:
        return self.distance_um is not None

    def release_images(self) -> None:
        self.rgb = None
        self.factin = None
        self.hnf4a = None
        self.dapi = None


def load_record_images(record: ImageRecord) -> None:
    composite_array = read_image_array(record.path)
    record.rgb = as_rgb(composite_array)
    height, width = record.rgb.shape[:2]
    channel_files = sibling_channel_paths(record.path)
    if channel_files:
        factin_path, hnf4a_path, dapi_path = channel_files
        record.factin = resize_float(as_intensity(read_image_array(factin_path)), (height, width))
        record.hnf4a = resize_float(as_intensity(read_image_array(hnf4a_path)), (height, width))
        record.dapi = resize_float(as_intensity(read_image_array(dapi_path)), (height, width))
    else:
        raw = np.asarray(composite_array)
        if raw.ndim == 2:
            gray = resize_float(as_intensity(raw), (height, width))
            record.factin = np.zeros_like(gray)
            record.hnf4a = gray.copy()
            record.dapi = gray.copy()
        else:
            raw_rgb = np.asarray(raw[..., :3], dtype=np.float32)
            raw_rgb = np.stack(
                [resize_float(raw_rgb[..., index], (height, width)) for index in range(3)],
                axis=-1,
            )
            record.hnf4a = raw_rgb[..., 0]
            record.factin = raw_rgb[..., 1]
            record.dapi = raw_rgb[..., 2]
    if record.pixel_source == "尚未读取":
        record.pixel_size_um, record.pixel_source = parse_pixel_size(record.path, width)


def composite_from_channels(
    record: ImageRecord,
    show_factin: bool,
    show_hnf4a: bool,
    show_dapi: bool,
) -> np.ndarray:
    assert record.rgb is not None
    if show_factin and show_hnf4a and show_dapi:
        return record.rgb
    assert record.factin is not None and record.hnf4a is not None and record.dapi is not None
    shape = record.factin.shape
    output = np.zeros((*shape, 3), dtype=np.uint8)
    if show_hnf4a:
        output[..., 0] = np.maximum(output[..., 0], normalize_uint8(record.hnf4a))
    if show_factin:
        factin = normalize_uint8(record.factin)
        output[..., 1] = np.maximum(output[..., 1], factin)
        output[..., 2] = np.maximum(output[..., 2], factin)
    if show_dapi:
        output[..., 2] = np.maximum(output[..., 2], normalize_uint8(record.dapi))
    return output


def analyze_record(
    record: ImageRecord,
    line_width_um: float,
    smoothing_sigma: float,
    background_percentile: float,
) -> None:
    if not record.loaded:
        load_record_images(record)
    if record.roi is None:
        raise ValueError("请先框选 ROI。")
    if record.line is None:
        raise ValueError("请先在 ROI 内画扫描线。")
    assert record.hnf4a is not None and record.dapi is not None
    x0, y0, width, height = record.roi
    x1, y1, x2, y2 = record.line
    if math.hypot(x2 - x1, y2 - y1) < 2:
        raise ValueError("扫描线太短，请重新绘制。")
    line_width_px = max(1, int(round(line_width_um / record.pixel_size_um)))
    hnf4a_profile = profile_line(
        record.hnf4a,
        (y1, x1),
        (y2, x2),
        linewidth=line_width_px,
        mode="constant",
        cval=0,
        reduce_func=np.mean,
    ).astype(float)
    dapi_profile = profile_line(
        record.dapi,
        (y1, x1),
        (y2, x2),
        linewidth=line_width_px,
        mode="constant",
        cval=0,
        reduce_func=np.mean,
    ).astype(float)
    if smoothing_sigma > 0:
        hnf4a_profile = gaussian_filter1d(hnf4a_profile, sigma=smoothing_sigma)
        dapi_profile = gaussian_filter1d(dapi_profile, sigma=smoothing_sigma)
    if background_percentile > 0:
        h_crop = record.hnf4a[y0 : y0 + height, x0 : x0 + width]
        d_crop = record.dapi[y0 : y0 + height, x0 : x0 + width]
        hnf4a_profile = np.clip(
            hnf4a_profile - np.percentile(h_crop, background_percentile), 0, None
        )
        dapi_profile = np.clip(
            dapi_profile - np.percentile(d_crop, background_percentile), 0, None
        )
    record.distance_um = np.linspace(
        0,
        math.hypot(x2 - x1, y2 - y1) * record.pixel_size_um,
        len(hnf4a_profile),
    )
    record.hnf4a_profile = hnf4a_profile
    record.dapi_profile = dapi_profile
    record.line_width_px = line_width_px
    record.analysis_line_width_um = line_width_um
    record.analysis_smoothing_sigma = smoothing_sigma
    record.analysis_background_percentile = background_percentile
    record.dirty = False


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
    dash: float,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    position = 0.0
    while position < length:
        end_position = min(position + dash, length)
        draw.line(
            (
                x1 + ux * position,
                y1 + uy * position,
                x1 + ux * end_position,
                y1 + uy * end_position,
            ),
            fill=fill,
            width=width,
        )
        position += dash * 1.75


def make_overlay(record: ImageRecord) -> Image.Image:
    assert record.rgb is not None and record.roi is not None and record.line is not None
    image = Image.fromarray(record.rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    x, y, width, height = record.roi
    stroke = max(3, int(round(image.width / 700)))
    draw.rectangle((x, y, x + width, y + height), outline=(255, 212, 0), width=stroke)
    x1, y1, x2, y2 = record.line
    draw_dashed_line(
        draw,
        (x1, y1),
        (x2, y2),
        fill=(255, 255, 255),
        width=max(2, stroke - 1),
        dash=max(8, image.width / 180),
    )
    radius = max(3, stroke + 1)
    for px, py in ((x1, y1), (x2, y2)):
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill="white")
    return image


def colorize_channel(intensity: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    gray = normalize_uint8(intensity).astype(np.float32) / 255.0
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    for index, component in enumerate(color):
        rgb[..., index] = np.round(gray * component).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def add_roi_scan_line(image: Image.Image, record: ImageRecord) -> Image.Image:
    assert record.roi is not None and record.line is not None
    output = image.copy().convert("RGB")
    x0, y0, _width, _height = record.roi
    x1, y1, x2, y2 = record.line
    relative_line = (x1 - x0, y1 - y0, x2 - x0, y2 - y0)
    draw = ImageDraw.Draw(output)
    stroke = max(2, int(round(output.width / 280)))
    dash = max(5, output.width / 35)
    draw_dashed_line(
        draw,
        (relative_line[0], relative_line[1]),
        (relative_line[2], relative_line[3]),
        fill=(255, 255, 255),
        width=stroke,
        dash=dash,
    )
    radius = max(2, stroke + 1)
    for px, py in (
        (relative_line[0], relative_line[1]),
        (relative_line[2], relative_line[3]),
    ):
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill="white")
    return output


def make_roi_channel_images(record: ImageRecord) -> dict[str, Image.Image]:
    assert record.rgb is not None
    assert record.factin is not None and record.hnf4a is not None and record.dapi is not None
    assert record.roi is not None
    x, y, width, height = record.roi
    composite = Image.fromarray(record.rgb[y : y + height, x : x + width], mode="RGB")
    factin = colorize_channel(record.factin[y : y + height, x : x + width], (0, 255, 255))
    signal1 = colorize_channel(record.hnf4a[y : y + height, x : x + width], (255, 0, 0))
    signal2 = colorize_channel(record.dapi[y : y + height, x : x + width], (0, 0, 255))
    return {
        "composite": composite,
        "factin": factin,
        "signal1": signal1,
        "signal2": signal2,
    }


def make_channel_panel(
    record: ImageRecord,
    roi_images: dict[str, Image.Image],
) -> Figure:
    panel = Figure(figsize=(8.0, 7.2), facecolor="white", constrained_layout=True)
    entries = [
        ("composite", "Composite", "black"),
        ("factin", "F-actin", "#008B8B"),
        ("signal1", record.signal1_name, HNF4A_COLOR),
        ("signal2", record.signal2_name, DAPI_COLOR),
    ]
    for index, (key, title, title_color) in enumerate(entries, start=1):
        axis = panel.add_subplot(2, 2, index)
        axis.imshow(add_roi_scan_line(roi_images[key], record))
        axis.set_axis_off()
        axis.set_title(title, color=title_color, fontsize=11, fontweight="bold")
    panel.suptitle(f"{record.name} — ROI channel view", fontsize=12)
    return panel


def make_curve_figure(record: ImageRecord) -> Figure:
    assert record.distance_um is not None
    assert record.hnf4a_profile is not None and record.dapi_profile is not None
    figure = Figure(figsize=(5.2, 3.5), facecolor="white", constrained_layout=True)
    axis = figure.add_subplot(111)
    axis.plot(
        record.distance_um,
        record.dapi_profile,
        color=DAPI_COLOR,
        lw=1.6,
        label=record.signal2_name,
    )
    axis.plot(
        record.distance_um,
        record.hnf4a_profile,
        color=HNF4A_COLOR,
        lw=1.6,
        label=record.signal1_name,
    )
    axis.set_xlim(0, max(1.0, float(record.distance_um[-1])))
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Distance (µm)")
    axis.set_ylabel("Fluorescence intensity (A.U.)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    return figure


def export_record(record: ImageRecord, output_dir: Path) -> dict[str, Path]:
    if not record.analyzed:
        raise ValueError("当前图像还没有完成分析。")
    if not record.loaded:
        load_record_images(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(record.path)
    signal1_file_label = safe_label(record.signal1_name)
    signal2_file_label = safe_label(record.signal2_name)
    csv_path = output_dir / f"{stem}_profile.csv"
    overlay_path = output_dir / f"{stem}_overlay.png"
    roi_composite_path = output_dir / f"{stem}_ROI_composite.png"
    roi_overlay_path = output_dir / f"{stem}_ROI_composite_overlay.png"
    roi_factin_path = output_dir / f"{stem}_ROI_F-actin.png"
    roi_signal1_path = output_dir / f"{stem}_ROI_{signal1_file_label}.png"
    roi_signal2_path = output_dir / f"{stem}_ROI_{signal2_file_label}.png"
    channel_panel_png = output_dir / f"{stem}_ROI_channels_panel.png"
    channel_panel_pdf = output_dir / f"{stem}_ROI_channels_panel.pdf"
    curve_png = output_dir / f"{stem}_curve.png"
    curve_pdf = output_dir / f"{stem}_curve.pdf"
    panel_png = output_dir / f"{stem}_analysis_panel.png"
    metadata_path = output_dir / f"{stem}_analysis.json"

    assert record.distance_um is not None
    assert record.hnf4a_profile is not None and record.dapi_profile is not None
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["distance_um", f"{record.signal1_name}_AU", f"{record.signal2_name}_AU"]
        )
        writer.writerows(
            zip(record.distance_um, record.hnf4a_profile, record.dapi_profile)
        )

    overlay = make_overlay(record)
    overlay.save(overlay_path, dpi=(300, 300))
    roi_images = make_roi_channel_images(record)
    roi_images["composite"].save(roi_composite_path, dpi=(300, 300))
    roi_images["factin"].save(roi_factin_path, dpi=(300, 300))
    roi_images["signal1"].save(roi_signal1_path, dpi=(300, 300))
    roi_images["signal2"].save(roi_signal2_path, dpi=(300, 300))
    roi_overlay = add_roi_scan_line(roi_images["composite"], record)
    roi_overlay.save(roi_overlay_path, dpi=(300, 300))
    channel_panel = make_channel_panel(record, roi_images)
    channel_panel.savefig(channel_panel_png, dpi=400, facecolor="white")
    channel_panel.savefig(channel_panel_pdf, facecolor="white")
    curve_figure = make_curve_figure(record)
    curve_figure.savefig(curve_png, dpi=400, facecolor="white")
    curve_figure.savefig(curve_pdf, facecolor="white")

    panel = Figure(figsize=(10.5, 4.4), facecolor="white", constrained_layout=True)
    image_axis = panel.add_subplot(121)
    curve_axis = panel.add_subplot(122)
    image_axis.imshow(roi_overlay)
    image_axis.set_axis_off()
    image_axis.set_title(f"{record.name} — ROI", fontsize=10)
    curve_axis.plot(
        record.distance_um,
        record.dapi_profile,
        color=DAPI_COLOR,
        lw=1.5,
        label=record.signal2_name,
    )
    curve_axis.plot(
        record.distance_um,
        record.hnf4a_profile,
        color=HNF4A_COLOR,
        lw=1.5,
        label=record.signal1_name,
    )
    curve_axis.set_xlim(0, max(1.0, float(record.distance_um[-1])))
    curve_axis.set_ylim(bottom=0)
    curve_axis.set_xlabel("Distance (µm)")
    curve_axis.set_ylabel("Fluorescence intensity (A.U.)")
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)
    curve_axis.legend(frameon=False)
    panel.savefig(panel_png, dpi=400, facecolor="white")

    metadata = {
        "source_image": str(record.path),
        "pixel_size_um": record.pixel_size_um,
        "pixel_size_source": record.pixel_source,
        "roi_xywh_px": list(record.roi or ()),
        "line_xyxy_full_image_px": list(record.line or ()),
        "line_xyxy_roi_relative_px": (
            [
                record.line[0] - record.roi[0],
                record.line[1] - record.roi[1],
                record.line[2] - record.roi[0],
                record.line[3] - record.roi[1],
            ]
            if record.line is not None and record.roi is not None
            else []
        ),
        "line_width_px": record.line_width_px,
        "line_width_um": record.analysis_line_width_um,
        "smoothing_sigma": record.analysis_smoothing_sigma,
        "background_percentile": record.analysis_background_percentile,
        "coordinate_system": "full-resolution image pixels",
        "signal_1_name": record.signal1_name,
        "signal_1_source": "c2/red channel",
        "signal_2_name": record.signal2_name,
        "signal_2_source": "c3/blue channel",
        "display_channel_note": "c1/cyan=F-actin when sibling channels exist",
        "note": "Exported pseudocolored TIFF intensities are descriptive A.U.; use raw CZI for rigorous between-sample quantification.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "csv": csv_path,
        "full_overlay": overlay_path,
        "roi_composite": roi_composite_path,
        "roi_overlay": roi_overlay_path,
        "roi_factin": roi_factin_path,
        "roi_signal1": roi_signal1_path,
        "roi_signal2": roi_signal2_path,
        "channel_panel_png": channel_panel_png,
        "channel_panel_pdf": channel_panel_pdf,
        "curve_png": curve_png,
        "curve_pdf": curve_pdf,
        "panel": panel_png,
        "metadata": metadata_path,
    }


class ImageCanvas(FigureCanvasQTAgg):
    selection_changed = Signal()
    message = Signal(str)
    mode_changed = Signal(str)

    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 7), facecolor="#111827")
        super().__init__(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_facecolor("#060A0F")
        self.figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96)
        self.record: ImageRecord | None = None
        self.show_factin = True
        self.show_hnf4a = True
        self.show_dapi = True
        self.selector: RectangleSelector | None = None
        self.line_connection: int | None = None
        self.line_points: list[tuple[float, float]] = []
        self.mode = "idle"
        self.roi_snapshot: tuple | None = None
        self.mpl_connect("key_press_event", self._on_key)
        self.render()

    def set_record(self, record: ImageRecord | None) -> None:
        self.cancel_mode()
        self.record = record
        self.render()

    def set_channels(self, factin: bool, hnf4a: bool, dapi: bool) -> None:
        if self.mode == "roi":
            self.finish_roi_edit(commit=True)
        elif self.mode == "line":
            self.cancel_mode()
        self.show_factin = factin
        self.show_hnf4a = hnf4a
        self.show_dapi = dapi
        self.render()

    def render(self) -> None:
        self.axis.clear()
        self.axis.set_facecolor("#060A0F")
        if self.record is None or not self.record.loaded:
            self.axis.text(
                0.5,
                0.5,
                "点击“导入图片”开始\n也可以把 TIFF/PNG 直接拖到窗口",
                transform=self.axis.transAxes,
                ha="center",
                va="center",
                color="#9CA3AF",
                fontsize=13,
                linespacing=1.7,
            )
            self.axis.set_axis_off()
            self.draw_idle()
            return
        display = composite_from_channels(
            self.record, self.show_factin, self.show_hnf4a, self.show_dapi
        )
        self.axis.imshow(display)
        self.axis.set_axis_off()
        self.axis.set_title(self.record.name, color="white", fontsize=11, pad=8)
        if self.record.roi is not None and self.mode != "roi":
            x, y, width, height = self.record.roi
            self.axis.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=False,
                    edgecolor=ROI_COLOR,
                    linewidth=2.2,
                )
            )
            self.axis.text(
                x,
                max(0, y - 8),
                f"ROI  {width * self.record.pixel_size_um:.1f} × "
                f"{height * self.record.pixel_size_um:.1f} µm",
                color=ROI_COLOR,
                fontsize=8,
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            )
        if self.record.line is not None:
            x1, y1, x2, y2 = self.record.line
            self.axis.plot(
                [x1, x2], [y1, y2], color=LINE_COLOR, lw=1.8, ls=(0, (5, 3))
            )
            self.axis.scatter(
                [x1, x2], [y1, y2], s=25, c="white", edgecolors="black", linewidths=0.6
            )
        if self.mode == "line" and self.line_points:
            px, py = self.line_points[0]
            self.axis.scatter([px], [py], s=35, c="white", edgecolors="black")
        self.draw_idle()

    def cancel_mode(self) -> None:
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector = None
        if self.line_connection is not None:
            self.mpl_disconnect(self.line_connection)
            self.line_connection = None
        self.mode = "idle"
        self.line_points.clear()
        self.mode_changed.emit("idle")

    def start_roi(self) -> None:
        if self.record is None or not self.record.loaded:
            self.message.emit("请先导入并选择一张图像。")
            return
        self.cancel_mode()
        self.roi_snapshot = (
            self.record.roi,
            self.record.line,
            self.record.distance_um,
            self.record.hnf4a_profile,
            self.record.dapi_profile,
            self.record.line_width_px,
            self.record.dirty,
        )
        self.record.line = None
        self.record.distance_um = None
        self.record.hnf4a_profile = None
        self.record.dapi_profile = None
        self.record.line_width_px = None
        self.record.dirty = True
        self.mode = "roi"
        self.render()
        self.selector = RectangleSelector(
            self.axis,
            self._roi_selected,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            drag_from_anywhere=True,
            props=dict(edgecolor=ROI_COLOR, facecolor="none", linewidth=2),
            handle_props=dict(
                marker="s",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=ROI_COLOR,
            ),
        )
        if self.record.roi is not None:
            x, y, width, height = self.record.roi
            self.selector.extents = (x, x + width, y, y + height)
            self.selector.set_visible(True)
            self.draw_idle()
        self.mode_changed.emit("roi")
        self.selection_changed.emit()
        self.message.emit(
            "ROI 编辑：拖动框内可移动，拖动边/角控制点可调整大小；Enter 完成，Esc 撤销。"
        )
        self.setFocus()

    def _roi_selected(self, start, end) -> None:
        if self.record is None or self.record.rgb is None:
            return
        if start.xdata is None or end.xdata is None or start.ydata is None or end.ydata is None:
            return
        x1, x2 = sorted((float(start.xdata), float(end.xdata)))
        y1, y2 = sorted((float(start.ydata), float(end.ydata)))
        image_height, image_width = self.record.rgb.shape[:2]
        x = int(np.clip(round(x1), 0, image_width - 2))
        y = int(np.clip(round(y1), 0, image_height - 2))
        width = int(np.clip(round(x2 - x1), 2, image_width - x))
        height = int(np.clip(round(y2 - y1), 2, image_height - y))
        self.record.roi = (x, y, width, height)
        self.record.line = None
        self.record.distance_um = None
        self.record.hnf4a_profile = None
        self.record.dapi_profile = None
        self.record.dirty = True
        if self.selector is not None:
            self.selector.extents = (x, x + width, y, y + height)
            self.selector.set_visible(True)
            self.draw_idle()
        self.selection_changed.emit()
        self.message.emit(
            "ROI 已更新；可以继续拖动/缩放，或按 Enter、点击“完成 ROI 编辑”。"
        )

    def finish_roi_edit(self, commit: bool = True) -> None:
        if self.mode != "roi":
            return
        record = self.record
        snapshot = self.roi_snapshot
        self.cancel_mode()
        if not commit and record is not None and snapshot is not None:
            (
                record.roi,
                record.line,
                record.distance_um,
                record.hnf4a_profile,
                record.dapi_profile,
                record.line_width_px,
                record.dirty,
            ) = snapshot
        self.roi_snapshot = None
        self.render()
        self.selection_changed.emit()
        if commit:
            self.message.emit("ROI 编辑完成。下一步画扫描线。")
        else:
            self.message.emit("已撤销本次 ROI 编辑。")

    def set_roi_size_um(self, width_um: float, height_um: float) -> None:
        if self.record is None or self.record.rgb is None:
            self.message.emit("请先导入并选择一张图像。")
            return
        image_height, image_width = self.record.rgb.shape[:2]
        width = max(2, min(int(round(width_um / self.record.pixel_size_um)), image_width))
        height = max(2, min(int(round(height_um / self.record.pixel_size_um)), image_height))
        if self.record.roi is None:
            center_x = image_width / 2
            center_y = image_height / 2
        else:
            x0, y0, old_width, old_height = self.record.roi
            center_x = x0 + old_width / 2
            center_y = y0 + old_height / 2
        x = int(np.clip(round(center_x - width / 2), 0, image_width - width))
        y = int(np.clip(round(center_y - height / 2), 0, image_height - height))
        self.record.roi = (x, y, width, height)
        self.record.line = None
        self.record.distance_um = None
        self.record.hnf4a_profile = None
        self.record.dapi_profile = None
        self.record.dirty = True
        if self.mode == "roi" and self.selector is not None:
            self.selector.extents = (x, x + width, y, y + height)
            self.selector.set_visible(True)
            self.draw_idle()
        else:
            self.render()
        self.selection_changed.emit()
        self.message.emit(
            f"ROI 尺寸已应用：{width * self.record.pixel_size_um:.2f} × "
            f"{height * self.record.pixel_size_um:.2f} µm。"
        )

    def start_line(self) -> None:
        if self.record is None or self.record.roi is None:
            self.message.emit("请先画 ROI。")
            return
        if self.mode == "roi":
            self.finish_roi_edit(commit=True)
        else:
            self.cancel_mode()
        self.mode = "line"
        self.line_points.clear()
        self.line_connection = self.mpl_connect("button_press_event", self._line_click)
        self.mode_changed.emit("line")
        self.message.emit("扫描线模式：在 ROI 内依次点击起点和终点。Esc 取消。")
        self.setFocus()

    def _line_click(self, event) -> None:
        if self.record is None or self.record.roi is None or event.inaxes is not self.axis:
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y, width, height = self.record.roi
        px, py = float(event.xdata), float(event.ydata)
        if not (x <= px <= x + width and y <= py <= y + height):
            self.message.emit("扫描线端点必须位于黄色 ROI 内。")
            return
        self.line_points.append((px, py))
        if len(self.line_points) == 2:
            p1, p2 = self.line_points
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 2:
                self.line_points.clear()
                self.message.emit("扫描线太短，请重新点击两个端点。")
                self.render()
                return
            self.record.line = (p1[0], p1[1], p2[0], p2[1])
            self.record.distance_um = None
            self.record.hnf4a_profile = None
            self.record.dapi_profile = None
            self.record.dirty = True
            self.cancel_mode()
            self.render()
            self.selection_changed.emit()
            self.message.emit("扫描线已设置。点击“③ 生成曲线”。")
        else:
            self.render()

    def _on_key(self, event) -> None:
        if self.mode == "roi" and event.key in ("enter", " "):
            self.finish_roi_edit(commit=True)
        elif self.mode == "roi" and event.key == "escape":
            self.finish_roi_edit(commit=False)
        elif event.key == "escape":
            self.cancel_mode()
            self.render()
            self.message.emit("已退出绘图模式。")


class CurveCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(4.8, 2.8), facecolor="white")
        super().__init__(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.18, right=0.97, bottom=0.30, top=0.93)
        self.setMinimumHeight(210)
        self.clear_plot()

    def clear_plot(self) -> None:
        self.axis.clear()
        self.axis.text(
            0.5,
            0.5,
            "完成 ROI 和扫描线后\n点击“生成曲线”",
            transform=self.axis.transAxes,
            ha="center",
            va="center",
            color="#6B7280",
            linespacing=1.6,
        )
        self.axis.set_axis_off()
        self.draw_idle()

    def show_record(self, record: ImageRecord | None) -> None:
        if record is None or not record.analyzed:
            self.clear_plot()
            return
        assert record.distance_um is not None
        assert record.hnf4a_profile is not None and record.dapi_profile is not None
        self.axis.clear()
        self.axis.plot(
            record.distance_um,
            record.dapi_profile,
            color=DAPI_COLOR,
            lw=1.5,
            label=record.signal2_name,
        )
        self.axis.plot(
            record.distance_um,
            record.hnf4a_profile,
            color=HNF4A_COLOR,
            lw=1.5,
            label=record.signal1_name,
        )
        self.axis.set_xlim(0, max(1.0, float(record.distance_um[-1])))
        self.axis.set_ylim(bottom=0)
        self.axis.set_xlabel("Distance (µm)")
        self.axis.set_ylabel("Fluorescence intensity (A.U.)")
        self.axis.spines["top"].set_visible(False)
        self.axis.spines["right"].set_visible(False)
        self.axis.legend(frameon=False, loc="upper right")
        self.axis.tick_params(labelsize=8)
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[ImageRecord] = []
        self.last_export_dir: Path | None = None
        self.setWindowTitle(APP_TITLE)
        self.resize(1550, 920)
        self.setMinimumSize(1180, 720)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_menu()
        self._apply_style()
        self._update_controls()

    @property
    def current_record(self) -> ImageRecord | None:
        row = self.image_list.currentRow()
        if 0 <= row < len(self.records):
            return self.records[row]
        return None

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        left_panel = QFrame()
        left_panel.setObjectName("sidePanel")
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(320)
        left_layout = QVBoxLayout(left_panel)
        title = QLabel("荧光线扫描")
        title.setObjectName("appTitle")
        subtitle = QLabel("导入 → 选区 → 画线 → 分析 → 导出")
        subtitle.setObjectName("subtitle")
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)

        import_row = QHBoxLayout()
        self.open_button = QPushButton("＋ 导入图片")
        self.folder_button = QPushButton("导入文件夹")
        import_row.addWidget(self.open_button)
        import_row.addWidget(self.folder_button)
        left_layout.addLayout(import_row)
        self.image_list = QListWidget()
        self.image_list.setAlternatingRowColors(True)
        self.image_list.setToolTip("绿色圆点表示已生成曲线；黄色圆点表示已画 ROI/扫描线")
        left_layout.addWidget(self.image_list, 1)
        list_row = QHBoxLayout()
        self.remove_button = QPushButton("移除")
        self.clear_button = QPushButton("清空列表")
        list_row.addWidget(self.remove_button)
        list_row.addWidget(self.clear_button)
        left_layout.addLayout(list_row)
        self.file_info = QLabel("尚未导入图像")
        self.file_info.setObjectName("infoBox")
        self.file_info.setWordWrap(True)
        left_layout.addWidget(self.file_info)
        splitter.addWidget(left_panel)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(6, 0, 6, 0)
        image_header = QHBoxLayout()
        image_header.addWidget(QLabel("图像与 Overlay"))
        image_header.addStretch(1)
        self.factin_check = QCheckBox("F-actin")
        self.hnf4a_check = QCheckBox("HNF4A")
        self.dapi_check = QCheckBox("DAPI")
        for checkbox in (self.factin_check, self.hnf4a_check, self.dapi_check):
            checkbox.setChecked(True)
            image_header.addWidget(checkbox)
        center_layout.addLayout(image_header)
        self.image_canvas = ImageCanvas()
        self.image_toolbar = NavigationToolbar2QT(self.image_canvas, center_panel)
        self.image_toolbar.setIconSize(self.image_toolbar.iconSize())
        center_layout.addWidget(self.image_toolbar)
        center_layout.addWidget(self.image_canvas, 1)
        self.canvas_hint = QLabel("提示：滚轮缩放；工具栏可平移/复位；ROI 为黄色，扫描线为白色虚线。")
        self.canvas_hint.setObjectName("hint")
        center_layout.addWidget(self.canvas_hint)
        splitter.addWidget(center_panel)

        right_panel = QFrame()
        right_panel.setObjectName("sidePanel")
        right_panel.setMinimumWidth(390)
        right_panel.setMaximumWidth(520)
        right_layout = QVBoxLayout(right_panel)

        parameters = QGroupBox("分析参数")
        parameter_form = QFormLayout(parameters)
        parameter_form.setContentsMargins(8, 8, 8, 6)
        parameter_form.setVerticalSpacing(3)
        self.pixel_spin = QDoubleSpinBox()
        self.pixel_spin.setRange(0.000001, 1000)
        self.pixel_spin.setDecimals(6)
        self.pixel_spin.setSuffix(" µm/px")
        self.pixel_spin.setValue(1.0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 50)
        self.width_spin.setDecimals(2)
        self.width_spin.setSuffix(" µm")
        self.width_spin.setValue(1.5)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0, 20)
        self.smooth_spin.setDecimals(1)
        self.smooth_spin.setValue(1.2)
        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0, 20)
        self.background_spin.setDecimals(1)
        self.background_spin.setSuffix(" %")
        self.background_spin.setValue(1.0)
        self.signal1_edit = QLineEdit("HNF4A")
        self.signal1_edit.setPlaceholderText("例如 HNF4A")
        self.signal2_edit = QLineEdit("DAPI")
        self.signal2_edit.setPlaceholderText("例如 DAPI")
        self.apply_names_all_button = QPushButton("将信号名称应用到全部图片")
        parameter_form.addRow("像素尺寸", self.pixel_spin)
        parameter_form.addRow("扫描带宽", self.width_spin)
        parameter_form.addRow("平滑 σ", self.smooth_spin)
        parameter_form.addRow("背景分位数", self.background_spin)
        parameter_form.addRow("信号 1（红/c2）", self.signal1_edit)
        parameter_form.addRow("信号 2（蓝/c3）", self.signal2_edit)
        parameter_form.addRow(self.apply_names_all_button)
        right_layout.addWidget(parameters)

        workflow = QGroupBox("操作流程")
        workflow_layout = QVBoxLayout(workflow)
        workflow_layout.setContentsMargins(8, 8, 8, 6)
        workflow_layout.setSpacing(3)
        roi_size_row = QHBoxLayout()
        roi_size_row.addWidget(QLabel("ROI 宽"))
        self.roi_width_spin = QDoubleSpinBox()
        self.roi_width_spin.setRange(1, 10000)
        self.roi_width_spin.setDecimals(2)
        self.roi_width_spin.setSuffix(" µm")
        self.roi_width_spin.setValue(90)
        roi_size_row.addWidget(self.roi_width_spin)
        roi_size_row.addWidget(QLabel("高"))
        self.roi_height_spin = QDoubleSpinBox()
        self.roi_height_spin.setRange(1, 10000)
        self.roi_height_spin.setDecimals(2)
        self.roi_height_spin.setSuffix(" µm")
        self.roi_height_spin.setValue(90)
        roi_size_row.addWidget(self.roi_height_spin)
        workflow_layout.addLayout(roi_size_row)
        self.apply_roi_size_button = QPushButton("应用输入的 ROI 长宽")
        workflow_layout.addWidget(self.apply_roi_size_button)
        self.roi_button = QPushButton("① 绘制 / 编辑 ROI")
        self.line_button = QPushButton("② 画扫描线（点击两个端点）")
        self.analyze_button = QPushButton("③ 生成曲线")
        self.analyze_button.setObjectName("primaryButton")
        workflow_layout.addWidget(self.roi_button)
        workflow_layout.addWidget(self.line_button)
        workflow_layout.addWidget(self.analyze_button)
        reset_row = QHBoxLayout()
        self.reset_roi_button = QPushButton("清除 ROI")
        self.reset_line_button = QPushButton("清除扫描线")
        reset_row.addWidget(self.reset_roi_button)
        reset_row.addWidget(self.reset_line_button)
        workflow_layout.addLayout(reset_row)
        right_layout.addWidget(workflow)

        curve_group = QGroupBox("曲线预览")
        curve_layout = QVBoxLayout(curve_group)
        self.curve_canvas = CurveCanvas()
        curve_layout.addWidget(self.curve_canvas)
        right_layout.addWidget(curve_group, 1)

        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout(export_group)
        export_row = QHBoxLayout()
        self.export_current_button = QPushButton("导出当前")
        self.export_all_button = QPushButton("批量导出全部")
        export_row.addWidget(self.export_current_button)
        export_row.addWidget(self.export_all_button)
        export_layout.addLayout(export_row)
        self.open_output_button = QPushButton("打开最近的导出文件夹")
        export_layout.addWidget(self.open_output_button)
        self.export_note = QLabel(
            "导出：全图定位、ROI 合并图、各通道 ROI、通道四宫格、曲线、CSV 和参数 JSON"
        )
        self.export_note.setObjectName("hint")
        self.export_note.setWordWrap(True)
        export_layout.addWidget(self.export_note)
        right_layout.addWidget(export_group)
        splitter.addWidget(right_panel)
        splitter.setSizes([270, 820, 430])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("准备就绪。请导入图像。")

        self.open_button.clicked.connect(self.choose_images)
        self.folder_button.clicked.connect(self.choose_folder)
        self.image_list.currentRowChanged.connect(self.select_record)
        self.remove_button.clicked.connect(self.remove_current)
        self.clear_button.clicked.connect(self.clear_records)
        self.factin_check.toggled.connect(self.update_channels)
        self.hnf4a_check.toggled.connect(self.update_channels)
        self.dapi_check.toggled.connect(self.update_channels)
        self.pixel_spin.valueChanged.connect(self.pixel_size_changed)
        self.width_spin.valueChanged.connect(self.parameters_changed)
        self.smooth_spin.valueChanged.connect(self.parameters_changed)
        self.background_spin.valueChanged.connect(self.parameters_changed)
        self.signal1_edit.editingFinished.connect(self.signal_names_changed)
        self.signal2_edit.editingFinished.connect(self.signal_names_changed)
        self.apply_names_all_button.clicked.connect(self.apply_signal_names_to_all)
        self.apply_roi_size_button.clicked.connect(self.apply_roi_size)
        self.roi_button.clicked.connect(self.begin_roi)
        self.line_button.clicked.connect(self.begin_line)
        self.analyze_button.clicked.connect(self.run_analysis)
        self.reset_roi_button.clicked.connect(self.clear_roi)
        self.reset_line_button.clicked.connect(self.clear_line)
        self.export_current_button.clicked.connect(self.export_current)
        self.export_all_button.clicked.connect(self.export_all)
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.image_canvas.selection_changed.connect(self.selection_changed)
        self.image_canvas.message.connect(self.statusBar().showMessage)
        self.image_canvas.mode_changed.connect(self.canvas_mode_changed)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_action = QAction("导入图片…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.choose_images)
        folder_action = QAction("导入文件夹…", self)
        folder_action.triggered.connect(self.choose_folder)
        export_action = QAction("导出当前…", self)
        export_action.setShortcut(QKeySequence.StandardKey.Save)
        export_action.triggered.connect(self.export_current)
        quit_action = QAction("退出", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addActions([open_action, folder_action, export_action])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        help_menu = self.menuBar().addMenu("帮助")
        about_action = QAction("使用说明", self)
        about_action.triggered.connect(self.show_help)
        help_menu.addAction(about_action)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { color: #111827; }
            QMainWindow { background: #F3F4F6; color: #111827; }
            QMenuBar { background: #F3F4F6; color: #111827; border-bottom: 1px solid #D1D5DB; }
            QMenuBar::item { background: transparent; color: #111827; padding: 5px 9px; }
            QMenuBar::item:selected { background: #E2E8F0; color: #111827; }
            QMenuBar::item:pressed { background: #CBD5E1; color: #111827; }
            QMenu { background: #FFFFFF; color: #111827; border: 1px solid #CBD5E1; }
            QMenu::item { color: #111827; padding: 6px 28px 6px 24px; }
            QMenu::item:selected { background: #DCEFEA; color: #134E4A; }
            QMenu::item:disabled { color: #9CA3AF; }
            QMenu::separator { height: 1px; background: #E5E7EB; margin: 4px 8px; }
            QFrame#sidePanel { background: white; border: 1px solid #D1D5DB; border-radius: 8px; }
            QLabel { color: #111827; background: transparent; }
            QLabel#appTitle { font-size: 20px; font-weight: 700; color: #111827; }
            QLabel#subtitle, QLabel#hint { color: #6B7280; font-size: 11px; }
            QLabel#infoBox { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 5px; padding: 8px; color: #374151; }
            QGroupBox { color: #111827; background: #FFFFFF; font-weight: 600; border: 1px solid #D1D5DB; border-radius: 7px; margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QCheckBox { color: #111827; background: transparent; spacing: 5px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QPushButton { color: #111827; min-height: 30px; border: 1px solid #CBD5E1; border-radius: 5px; background: #FFFFFF; padding: 3px 8px; }
            QPushButton:hover { color: #111827; background: #F1F5F9; border-color: #94A3B8; }
            QPushButton:pressed { color: #111827; background: #E2E8F0; }
            QPushButton:disabled { color: #9CA3AF; background: #F3F4F6; }
            QPushButton#primaryButton { background: #0F766E; color: white; border-color: #0F766E; font-weight: 700; min-height: 36px; }
            QPushButton#primaryButton:hover { background: #115E59; }
            QListWidget { color: #111827; background: #FFFFFF; alternate-background-color: #F8FAFC; border: 1px solid #D1D5DB; border-radius: 5px; padding: 3px; }
            QListWidget::item { min-height: 29px; padding: 3px; }
            QListWidget::item:selected { background: #DCEFEA; color: #134E4A; }
            QDoubleSpinBox, QLineEdit { color: #111827; background: #FFFFFF; selection-background-color: #0F766E; selection-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 3px; padding: 2px 5px; min-height: 26px; }
            QDoubleSpinBox:disabled, QLineEdit:disabled { color: #9CA3AF; background: #F3F4F6; }
            QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { background: #F1F5F9; border-left: 1px solid #CBD5E1; width: 18px; }
            QToolBar { background: #F3F4F6; color: #111827; border: none; spacing: 3px; }
            QToolButton { color: #111827; background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 3px; }
            QToolButton:hover { color: #111827; background: #E2E8F0; border-color: #CBD5E1; }
            QToolButton:checked { color: #111827; background: #CBD5E1; }
            QStatusBar { color: #111827; background: #F3F4F6; border-top: 1px solid #D1D5DB; }
            QStatusBar QLabel { color: #111827; }
            QToolTip { color: #111827; background: #FFFFFF; border: 1px solid #94A3B8; padding: 3px; }
            """
        )

    def _update_controls(self) -> None:
        record = self.current_record
        has_record = record is not None
        has_roi = bool(record and record.roi)
        has_line = bool(record and record.line)
        analyzed = bool(record and record.analyzed)
        for widget in (
            self.pixel_spin,
            self.width_spin,
            self.smooth_spin,
            self.background_spin,
            self.signal1_edit,
            self.signal2_edit,
            self.apply_names_all_button,
            self.roi_width_spin,
            self.roi_height_spin,
            self.apply_roi_size_button,
            self.roi_button,
        ):
            widget.setEnabled(has_record)
        self.line_button.setEnabled(has_roi)
        self.analyze_button.setEnabled(has_line)
        self.reset_roi_button.setEnabled(has_roi)
        self.reset_line_button.setEnabled(has_line)
        self.export_current_button.setEnabled(analyzed)
        self.export_all_button.setEnabled(any(item.analyzed for item in self.records))
        self.open_output_button.setEnabled(self.last_export_dir is not None)
        self.remove_button.setEnabled(has_record)
        self.clear_button.setEnabled(bool(self.records))

    def _refresh_list_item(self, index: int) -> None:
        if not (0 <= index < len(self.records)):
            return
        record = self.records[index]
        item = self.image_list.item(index)
        if record.analyzed:
            marker = "●"
            color = "#059669"
            state = "已分析"
        elif record.line:
            marker = "●"
            color = "#D97706"
            state = "已画线"
        elif record.roi:
            marker = "●"
            color = "#D97706"
            state = "已有 ROI"
        else:
            marker = "○"
            color = "#6B7280"
            state = "未选择"
        item.setText(f"{marker}  {record.name}")
        item.setForeground(QColor(color))
        item.setToolTip(f"{record.path}\n状态：{state}")

    def add_paths(self, paths: list[Path]) -> None:
        existing = {record.path.resolve() for record in self.records}
        added = 0
        for path in paths:
            path = path.resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if path in existing:
                continue
            record = ImageRecord(
                path=path,
                name=display_name(path),
                signal1_name=self.signal1_edit.text().strip() or "Signal 1",
                signal2_name=self.signal2_edit.text().strip() or "Signal 2",
            )
            self.records.append(record)
            self.image_list.addItem(QListWidgetItem(record.name))
            self._refresh_list_item(len(self.records) - 1)
            existing.add(path)
            added += 1
        if added and self.image_list.currentRow() < 0:
            self.image_list.setCurrentRow(0)
        self.statusBar().showMessage(f"已导入 {added} 张图像；列表共 {len(self.records)} 张。")
        self._update_controls()

    def choose_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择显微镜图像",
            str(Path.cwd()),
            "图像文件 (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;所有文件 (*)",
        )
        if files:
            self.add_paths([Path(path) for path in files])

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", str(Path.cwd()))
        if not folder:
            return
        root = Path(folder)
        composites = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".tif", ".tiff"}
            and (path.name.endswith("c1-3.tif") or path.name.endswith("c1-3.tiff"))
        )
        if not composites:
            composites = sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
        self.add_paths(composites)

    def select_record(self, row: int) -> None:
        if not (0 <= row < len(self.records)):
            self.image_canvas.set_record(None)
            self.curve_canvas.clear_plot()
            self.file_info.setText("尚未导入图像")
            self._update_controls()
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for index, other in enumerate(self.records):
                if index != row and other.loaded:
                    other.release_images()
            record = self.records[row]
            if not record.loaded:
                load_record_images(record)
            self.pixel_spin.blockSignals(True)
            self.pixel_spin.setValue(record.pixel_size_um)
            self.pixel_spin.blockSignals(False)
            self.signal1_edit.blockSignals(True)
            self.signal2_edit.blockSignals(True)
            self.signal1_edit.setText(record.signal1_name)
            self.signal2_edit.setText(record.signal2_name)
            self.signal1_edit.blockSignals(False)
            self.signal2_edit.blockSignals(False)
            self.hnf4a_check.setText(record.signal1_name)
            self.dapi_check.setText(record.signal2_name)
            self.image_canvas.set_record(record)
            self.curve_canvas.show_record(record)
            self.sync_roi_inputs()
            self.update_file_info()
            self.statusBar().showMessage(f"已载入：{record.path}")
        except Exception as error:
            QMessageBox.critical(self, "无法读取图像", f"{self.records[row].path}\n\n{error}")
        finally:
            QApplication.restoreOverrideCursor()
            self._update_controls()

    def remove_current(self) -> None:
        row = self.image_list.currentRow()
        if 0 <= row < len(self.records):
            self.records.pop(row)
            self.image_list.takeItem(row)
            if self.records:
                self.image_list.setCurrentRow(min(row, len(self.records) - 1))
            else:
                self.image_canvas.set_record(None)
                self.curve_canvas.clear_plot()
            self._update_controls()

    def clear_records(self) -> None:
        if self.records and QMessageBox.question(
            self, "清空列表", "要清空所有已导入图像和当前选择吗？"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.records.clear()
        self.image_list.clear()
        self.image_canvas.set_record(None)
        self.curve_canvas.clear_plot()
        self.file_info.setText("尚未导入图像")
        self._update_controls()

    def update_channels(self) -> None:
        self.image_canvas.set_channels(
            self.factin_check.isChecked(),
            self.hnf4a_check.isChecked(),
            self.dapi_check.isChecked(),
        )

    def update_file_info(self) -> None:
        record = self.current_record
        if record is None or record.rgb is None:
            self.file_info.setText("尚未导入图像")
            return
        height, width = record.rgb.shape[:2]
        channel_note = (
            f"独立通道：c2={record.signal1_name}，c3={record.signal2_name}"
            if sibling_channel_paths(record.path)
            else f"RGB：红={record.signal1_name}，蓝={record.signal2_name}"
        )
        self.file_info.setText(
            f"{record.name}\n{width} × {height} px\n"
            f"{record.pixel_size_um:.6f} µm/px\n{record.pixel_source}\n{channel_note}"
        )

    def signal_names_changed(self) -> None:
        record = self.current_record
        if record is None:
            return
        signal1 = self.signal1_edit.text().strip() or "Signal 1"
        signal2 = self.signal2_edit.text().strip() or "Signal 2"
        self.signal1_edit.setText(signal1)
        self.signal2_edit.setText(signal2)
        record.signal1_name = signal1
        record.signal2_name = signal2
        self.hnf4a_check.setText(signal1)
        self.dapi_check.setText(signal2)
        self.curve_canvas.show_record(record)
        self.update_file_info()
        self.statusBar().showMessage(
            f"信号名称已更新：红/c2 = {signal1}；蓝/c3 = {signal2}。"
        )

    def apply_signal_names_to_all(self) -> None:
        self.signal_names_changed()
        signal1 = self.signal1_edit.text().strip() or "Signal 1"
        signal2 = self.signal2_edit.text().strip() or "Signal 2"
        for record in self.records:
            record.signal1_name = signal1
            record.signal2_name = signal2
        self.curve_canvas.show_record(self.current_record)
        self.statusBar().showMessage(
            f"已将信号名称应用到全部 {len(self.records)} 张图片。"
        )

    def sync_roi_inputs(self) -> None:
        record = self.current_record
        if record is None or record.roi is None:
            return
        _x, _y, width, height = record.roi
        self.roi_width_spin.blockSignals(True)
        self.roi_height_spin.blockSignals(True)
        self.roi_width_spin.setValue(width * record.pixel_size_um)
        self.roi_height_spin.setValue(height * record.pixel_size_um)
        self.roi_width_spin.blockSignals(False)
        self.roi_height_spin.blockSignals(False)

    def apply_roi_size(self) -> None:
        self.image_canvas.set_roi_size_um(
            self.roi_width_spin.value(), self.roi_height_spin.value()
        )

    def canvas_mode_changed(self, mode: str) -> None:
        if mode == "roi":
            self.roi_button.setText("完成 ROI 编辑（Enter）")
            self.roi_button.setObjectName("primaryButton")
        else:
            self.roi_button.setText("① 绘制 / 编辑 ROI")
            self.roi_button.setObjectName("")
        self.roi_button.style().unpolish(self.roi_button)
        self.roi_button.style().polish(self.roi_button)

    def pixel_size_changed(self, value: float) -> None:
        record = self.current_record
        if record is None:
            return
        record.pixel_size_um = value
        record.pixel_source = "用户在 UI 中设置"
        record.distance_um = None
        record.hnf4a_profile = None
        record.dapi_profile = None
        record.dirty = True
        self.image_canvas.render()
        self.sync_roi_inputs()
        self.update_file_info()
        self.curve_canvas.clear_plot()
        self._refresh_list_item(self.image_list.currentRow())
        self._update_controls()

    def parameters_changed(self) -> None:
        record = self.current_record
        if record is not None and record.analyzed:
            record.distance_um = None
            record.hnf4a_profile = None
            record.dapi_profile = None
            record.dirty = True
            self.curve_canvas.clear_plot()
            self._refresh_list_item(self.image_list.currentRow())
            self.statusBar().showMessage("参数已改变，请重新点击“生成曲线”。")
            self._update_controls()

    def begin_roi(self) -> None:
        if self.image_canvas.mode == "roi":
            self.image_canvas.finish_roi_edit(commit=True)
        else:
            self.image_canvas.start_roi()

    def begin_line(self) -> None:
        self.image_canvas.start_line()

    def selection_changed(self) -> None:
        row = self.image_list.currentRow()
        self._refresh_list_item(row)
        self.sync_roi_inputs()
        self.curve_canvas.show_record(self.current_record)
        self._update_controls()

    def clear_roi(self) -> None:
        record = self.current_record
        if record is None:
            return
        record.roi = None
        record.line = None
        record.distance_um = None
        record.hnf4a_profile = None
        record.dapi_profile = None
        record.dirty = True
        self.image_canvas.cancel_mode()
        self.image_canvas.render()
        self.selection_changed()
        self.statusBar().showMessage("ROI 和扫描线已清除。")

    def clear_line(self) -> None:
        record = self.current_record
        if record is None:
            return
        record.line = None
        record.distance_um = None
        record.hnf4a_profile = None
        record.dapi_profile = None
        record.dirty = True
        self.image_canvas.cancel_mode()
        self.image_canvas.render()
        self.selection_changed()
        self.statusBar().showMessage("扫描线已清除。")

    def run_analysis(self, checked: bool = False, silent: bool = False) -> bool:
        del checked
        record = self.current_record
        if record is None:
            return False
        try:
            analyze_record(
                record,
                self.width_spin.value(),
                self.smooth_spin.value(),
                self.background_spin.value(),
            )
            self.curve_canvas.show_record(record)
            self._refresh_list_item(self.image_list.currentRow())
            self._update_controls()
            self.statusBar().showMessage(
                f"曲线已生成：{len(record.distance_um) if record.distance_um is not None else 0} 个采样点，"
                f"扫描带宽 {record.line_width_px} px。"
            )
            return True
        except Exception as error:
            if not silent:
                QMessageBox.warning(self, "无法分析", str(error))
            return False

    def _choose_export_dir(self) -> Path | None:
        initial = self.last_export_dir or Path.cwd()
        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(initial))
        return Path(folder) if folder else None

    def export_current(self) -> None:
        record = self.current_record
        if record is None or not record.analyzed:
            QMessageBox.information(self, "尚未分析", "请先生成当前图像的曲线。")
            return
        output_dir = self._choose_export_dir()
        if output_dir is None:
            return
        try:
            paths = export_record(record, output_dir)
            self.last_export_dir = output_dir
            self._update_controls()
            self.statusBar().showMessage(f"已导出到：{output_dir}")
            QMessageBox.information(
                self,
                "导出完成",
                f"已生成 {len(paths)} 个文件，包括 ROI 裁剪图、各通道图、"
                f"通道组合图、曲线和数据。\n\n{output_dir}",
            )
        except Exception as error:
            QMessageBox.critical(self, "导出失败", str(error))

    def export_all(self) -> None:
        analyzed = [record for record in self.records if record.analyzed]
        if not analyzed:
            QMessageBox.information(self, "没有可导出的结果", "请至少完成一张图像的分析。")
            return
        output_dir = self._choose_export_dir()
        if output_dir is None:
            return
        progress = QProgressDialog("正在批量导出…", "取消", 0, len(analyzed), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        errors: list[str] = []
        for index, record in enumerate(analyzed):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"正在导出：{record.name}")
            QApplication.processEvents()
            try:
                load_record_images(record)
                export_record(record, output_dir)
            except Exception as error:
                errors.append(f"{record.name}: {error}")
            finally:
                if record is not self.current_record:
                    record.release_images()
            progress.setValue(index + 1)
        self.last_export_dir = output_dir
        self._update_controls()
        if errors:
            QMessageBox.warning(self, "批量导出完成（有错误）", "\n".join(errors))
        else:
            QMessageBox.information(self, "批量导出完成", f"结果已保存到：\n{output_dir}")

    def open_output_folder(self) -> None:
        if self.last_export_dir is None:
            return
        try:
            os.startfile(self.last_export_dir)  # type: ignore[attr-defined]
        except AttributeError:
            subprocess.Popen(["explorer", str(self.last_export_dir)])

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "使用说明",
            "1. 导入合并 TIFF（若同目录存在 c1/c2/c3，软件会自动读取）。\n"
            "2. 确认像素尺寸，并输入红/c2、蓝/c3信号的名称。\n"
            "3. 点击“绘制/编辑 ROI”；拖动框内移动，拖动边角调整大小。\n"
            "4. 需要精确尺寸时输入 ROI 宽和高，再点击“应用输入的 ROI 长宽”。\n"
            "5. 点击“画扫描线”，在 ROI 内点击两个端点，然后生成曲线。\n"
            "6. 导出全图定位、ROI 裁剪、各通道 ROI、曲线图和 CSV。\n\n"
            "提示：当前 ZEISS 导出的 8-bit 伪彩色通道适合展示性线扫描；"
            "严格的样本间强度比较应使用原始 CZI/16-bit 数据。",
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths: list[Path] = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                composites = sorted(path.rglob("*c1-3.tif"))
                paths.extend(composites or [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES])
            else:
                paths.append(path)
        self.add_paths(paths)
        event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:
        event.accept()


def run_self_test(image_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    configure_light_theme(app)
    configure_fonts(app)
    window = MainWindow()
    window.add_paths([image_path])
    record = window.current_record
    if record is None or record.rgb is None:
        raise RuntimeError("self-test could not load the image")
    window.signal1_edit.setText("Marker A")
    window.signal2_edit.setText("Nuclear stain")
    window.signal_names_changed()
    window.roi_width_spin.setValue(70.0)
    window.roi_height_spin.setValue(50.0)
    window.apply_roi_size()
    if record.roi is None:
        raise RuntimeError("self-test ROI size input failed")
    _rx, _ry, input_width_px, input_height_px = record.roi
    if abs(input_width_px * record.pixel_size_um - 70.0) > record.pixel_size_um:
        raise RuntimeError("self-test ROI width input was not applied")
    if abs(input_height_px * record.pixel_size_um - 50.0) > record.pixel_size_um:
        raise RuntimeError("self-test ROI height input was not applied")
    window.image_canvas.start_roi()
    x, y, roi_width, roi_height = record.roi
    window.image_canvas._roi_selected(
        SimpleNamespace(xdata=x + 5, ydata=y + 5),
        SimpleNamespace(xdata=x + roi_width - 5, ydata=y + roi_height - 5),
    )
    if window.image_canvas.mode != "roi":
        raise RuntimeError("self-test editable ROI did not remain active")
    window.image_canvas.finish_roi_edit(commit=True)
    x, y, roi_width, roi_height = record.roi
    record.line = (
        x + roi_width * 0.1,
        y + roi_height * 0.5,
        x + roi_width * 0.9,
        y + roi_height * 0.5,
    )
    if not window.run_analysis(silent=True):
        raise RuntimeError("self-test analysis failed")
    with tempfile.TemporaryDirectory(prefix="hnf4a_gui_test_") as folder:
        exported = export_record(record, Path(folder))
        missing = [path for path in exported.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"self-test export missing: {missing}")
        csv_header = exported["csv"].read_text(encoding="utf-8-sig").splitlines()[0]
        metadata = json.loads(exported["metadata"].read_text(encoding="utf-8"))
        if "Marker A_AU" not in csv_header or "Nuclear stain_AU" not in csv_header:
            raise RuntimeError("self-test custom signal names missing from CSV")
        if metadata.get("signal_1_name") != "Marker A":
            raise RuntimeError("self-test custom signal names missing from metadata")
        expected_size = (record.roi[2], record.roi[3])
        for key in ("roi_composite", "roi_overlay", "roi_factin", "roi_signal1", "roi_signal2"):
            with Image.open(exported[key]) as roi_image:
                if roi_image.size != expected_size:
                    raise RuntimeError(
                        f"self-test {key} size {roi_image.size} != ROI {expected_size}"
                    )
        print(f"Self-test passed: {len(exported)} output files")
    window.close()
    app.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("images", nargs="*", type=Path, help="images to open")
    parser.add_argument("--self-test", type=Path, help="run an offscreen analysis/export test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.self_test)
        return
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    configure_light_theme(app)
    configure_fonts(app)
    window = MainWindow()
    if args.images:
        window.add_paths(args.images)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
