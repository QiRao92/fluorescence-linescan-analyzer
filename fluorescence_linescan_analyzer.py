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
from dataclasses import dataclass, field
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
from PySide6.QtCore import QPoint, Qt, Signal
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
    QPainter,
    QPolygon,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QProxyStyle,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter1d
from skimage.measure import profile_line


APP_TITLE = "Fluorescence Line-scan Analyzer"
ROI_COLOR = "#FFD400"
LINE_COLOR = "#FFFFFF"
SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
CHANNEL_COLORS = (
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7A3E9D",
    "#666666",
)


class VisibleSpinArrowStyle(QProxyStyle):
    """Draw spin-box arrows explicitly so Windows theme colors cannot hide them."""

    def drawPrimitive(self, element, option, painter, widget=None) -> None:  # noqa: N802
        spin_elements = (
            QStyle.PrimitiveElement.PE_IndicatorSpinUp,
            QStyle.PrimitiveElement.PE_IndicatorSpinDown,
        )
        if element not in spin_elements:
            super().drawPrimitive(element, option, painter, widget)
            return
        rect = option.rect
        half_width = max(3, min(rect.width(), rect.height()) // 4)
        center_x = rect.center().x()
        center_y = rect.center().y()
        if element == QStyle.PrimitiveElement.PE_IndicatorSpinUp:
            points = QPolygon(
                [
                    QPoint(center_x - half_width, center_y + 2),
                    QPoint(center_x + half_width, center_y + 2),
                    QPoint(center_x, center_y - half_width + 1),
                ]
            )
        else:
            points = QPolygon(
                [
                    QPoint(center_x - half_width, center_y - 2),
                    QPoint(center_x + half_width, center_y - 2),
                    QPoint(center_x, center_y + half_width - 1),
                ]
            )
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        painter.setBrush(QColor("#334155" if enabled else "#94A3B8"))
        painter.drawPolygon(points)
        painter.restore()


def set_color_button_style(button: QPushButton, color: str) -> None:
    qt_color = QColor(color)
    luminance = 0.299 * qt_color.red() + 0.587 * qt_color.green() + 0.114 * qt_color.blue()
    text_color = "#111827" if luminance > 150 else "#FFFFFF"
    button.setText(f"颜色  {color.upper()}")
    button.setStyleSheet(
        "QPushButton {"
        f"background: {color}; color: {text_color}; border: 1px solid #64748B;"
        "font-weight: 700; min-height: 28px; padding: 2px 8px;"
        "}"
    )


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
    app.setStyle(VisibleSpinArrowStyle("Fusion"))
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
    array = np.squeeze(array)
    while array.ndim > 3:
        array = array[0]
    if (
        array.ndim == 3
        and 1 <= array.shape[0] <= 16
        and array.shape[-1] > 16
    ):
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


def sibling_channel_paths(composite: Path) -> list[Path]:
    name = composite.name
    series_match = re.match(r"^(.*)_s1c1-3\.tiff?$", name, re.IGNORECASE)
    channel_match = re.match(r"^(.*)_c1-3\.tiff?$", name, re.IGNORECASE)
    if series_match:
        prefix = series_match.group(1)
        expression = re.compile(rf"^{re.escape(prefix)}_s1c(\d+)\.tiff?$", re.IGNORECASE)
    elif channel_match:
        prefix = channel_match.group(1)
        expression = re.compile(rf"^{re.escape(prefix)}_c(\d+)\.tiff?$", re.IGNORECASE)
    else:
        return []
    matches: list[tuple[int, Path]] = []
    for path in composite.parent.iterdir():
        match = expression.match(path.name)
        if match and path.is_file():
            matches.append((int(match.group(1)), path))
    return [path for _index, path in sorted(matches)]


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
class SourceChannel:
    key: str
    label: str
    color: str
    data: np.ndarray
    source_path: Path | None = None


@dataclass
class AnalysisChannel:
    source_key: str
    name: str = ""
    color: str = "#0072B2"
    profile: np.ndarray | None = field(default=None, repr=False)
    profile_sd: np.ndarray | None = field(default=None, repr=False)


class ChannelEditorDialog(QDialog):
    def __init__(
        self,
        sources: list[SourceChannel],
        used_source_keys: set[str],
        existing: AnalysisChannel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑分析通道" if existing else "添加分析通道")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.selected_color = existing.color if existing else "#0072B2"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.source_combo = QComboBox()
        for source in sources:
            if source.key not in used_source_keys or (
                existing is not None and source.key == existing.source_key
            ):
                self.source_combo.addItem(source.label, source.key)
        self.name_edit = QLineEdit(existing.name if existing else "")
        self.name_edit.setPlaceholderText("输入通道名称")
        self.color_button = QPushButton()
        form.addRow("图像通道来源", self.source_combo)
        form.addRow("通道名称", self.name_edit)
        form.addRow("显示颜色", self.color_button)
        layout.addLayout(form)

        hint = QLabel("该颜色将同步用于主图伪彩色、曲线和导出图。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        layout.addWidget(buttons)

        if existing is not None:
            existing_index = self.source_combo.findData(existing.source_key)
            self.source_combo.setCurrentIndex(max(0, existing_index))
        elif self.source_combo.count():
            first_source = next(
                (source for source in sources if source.key == self.source_combo.currentData()),
                None,
            )
            if first_source is not None:
                self.selected_color = first_source.color
        set_color_button_style(self.color_button, self.selected_color)

        self.source_combo.currentIndexChanged.connect(self.source_changed)
        self.color_button.clicked.connect(self.choose_color)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)

    def source_changed(self, _index: int = -1) -> None:
        source_key = str(self.source_combo.currentData())
        parent = self.parent()
        if isinstance(parent, MainWindow):
            record = parent.current_record
            source = record.source_channel(source_key) if record else None
            if source is not None:
                self.selected_color = source.color
                set_color_button_style(self.color_button, self.selected_color)

    def choose_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self.selected_color), self, "选择分析通道颜色"
        )
        if selected.isValid():
            self.selected_color = selected.name().upper()
            set_color_button_style(self.color_button, self.selected_color)

    def validate_and_accept(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "缺少通道名称", "请输入通道名称。")
            self.name_edit.setFocus()
            return
        self.accept()

    def values(self) -> tuple[str, str, str]:
        return (
            str(self.source_combo.currentData()),
            self.name_edit.text().strip(),
            self.selected_color,
        )


@dataclass
class ImageRecord:
    path: Path
    name: str
    pixel_size_um: float = 1.0
    pixel_source: str = "尚未读取"
    rgb: np.ndarray | None = None
    source_channels: list[SourceChannel] = field(default_factory=list)
    display_source_keys: set[str] = field(default_factory=set)
    analysis_channels: list[AnalysisChannel] = field(default_factory=list)
    roi: tuple[int, int, int, int] | None = None
    line: tuple[float, float, float, float] | None = None
    distance_um: np.ndarray | None = None
    line_width_px: int | None = None
    analysis_line_width_um: float | None = None
    analysis_smoothing_sigma: float | None = None
    analysis_background_percentile: float | None = None
    analysis_compute_sd: bool = False
    dirty: bool = True

    @property
    def loaded(self) -> bool:
        return self.rgb is not None and bool(self.source_channels)

    @property
    def analyzed(self) -> bool:
        return (
            self.distance_um is not None
            and bool(self.analysis_channels)
            and all(channel.profile is not None for channel in self.analysis_channels)
        )

    def source_channel(self, key: str) -> SourceChannel | None:
        return next((channel for channel in self.source_channels if channel.key == key), None)

    def invalidate_analysis(self) -> None:
        self.distance_um = None
        for channel in self.analysis_channels:
            channel.profile = None
            channel.profile_sd = None
        self.line_width_px = None
        self.dirty = True

    def release_images(self) -> None:
        self.rgb = None
        self.source_channels.clear()


def _source_color(index: int, count: int, rgb_source: bool = False) -> str:
    if rgb_source and count >= 3:
        return ("#FF0000", "#00C853", "#0066FF")[index] if index < 3 else CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
    if count == 3:
        return ("#00D5E8", "#FF3B30", "#0066FF")[index]
    return CHANNEL_COLORS[index % len(CHANNEL_COLORS)]


def _array_channels(array: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(array)
    if values.ndim == 2:
        return [np.asarray(values, dtype=np.float32)]
    if values.ndim != 3:
        raise ValueError(f"不支持的图像维度：{values.shape}")
    return [np.asarray(values[..., index], dtype=np.float32) for index in range(values.shape[-1])]


def load_record_images(record: ImageRecord) -> None:
    previous_display_keys = set(record.display_source_keys)
    composite_array = read_image_array(record.path)
    record.rgb = as_rgb(composite_array)
    height, width = record.rgb.shape[:2]
    source_channels: list[SourceChannel] = []
    channel_files = sibling_channel_paths(record.path)
    if channel_files:
        count = len(channel_files)
        for index, channel_path in enumerate(channel_files):
            channel_number_match = re.search(r"c(\d+)\.tiff?$", channel_path.name, re.IGNORECASE)
            channel_number = channel_number_match.group(1) if channel_number_match else str(index + 1)
            source_channels.append(
                SourceChannel(
                    key=f"c{channel_number}",
                    label=f"c{channel_number}",
                    color=_source_color(index, count),
                    data=resize_float(
                        as_intensity(read_image_array(channel_path)), (height, width)
                    ),
                    source_path=channel_path,
                )
            )
    else:
        raw_channels = _array_channels(composite_array)
        count = len(raw_channels)
        rgb_labels = ("R（红）", "G（绿）", "B（蓝）")
        for index, values in enumerate(raw_channels):
            if count == 3:
                key = ("R", "G", "B")[index]
                label = rgb_labels[index]
            elif count == 1:
                key = "Gray"
                label = "灰度通道"
            else:
                key = f"Ch{index + 1}"
                label = f"通道 {index + 1}"
            source_channels.append(
                SourceChannel(
                    key=key,
                    label=label,
                    color="#FFFFFF" if count == 1 else _source_color(index, count, rgb_source=True),
                    data=resize_float(values, (height, width)),
                    source_path=record.path,
                )
            )
    record.source_channels = source_channels
    valid_keys = {channel.key for channel in source_channels}
    record.display_source_keys = (
        previous_display_keys & valid_keys if previous_display_keys & valid_keys else valid_keys
    )
    record.analysis_channels = [
        channel for channel in record.analysis_channels if channel.source_key in valid_keys
    ]
    if record.pixel_source == "尚未读取":
        record.pixel_size_um, record.pixel_source = parse_pixel_size(record.path, width)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def composite_from_channels(
    record: ImageRecord,
    visible_source_keys: set[str] | None = None,
) -> np.ndarray:
    assert record.rgb is not None
    selected = visible_source_keys if visible_source_keys is not None else record.display_source_keys
    height, width = record.rgb.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in record.source_channels:
        if channel.key not in selected:
            continue
        intensity = normalize_uint8(channel.data).astype(np.float32) / 255.0
        color = hex_to_rgb(channel.color)
        for component, value in enumerate(color):
            contribution = np.round(intensity * value).astype(np.uint8)
            output[..., component] = np.maximum(output[..., component], contribution)
    return output


def validate_analysis_channels(record: ImageRecord) -> None:
    if not record.analysis_channels:
        raise ValueError("请先点击“添加分析通道”，选择来源并输入信号名称。")
    names = [channel.name.strip() for channel in record.analysis_channels]
    if any(not name for name in names):
        raise ValueError("每个分析通道都必须由操作者输入信号名称。")
    if len(set(names)) != len(names):
        raise ValueError("分析通道名称不能重复。")
    source_keys = [channel.source_key for channel in record.analysis_channels]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("同一个图像通道只能添加一次。")
    if any(record.source_channel(key) is None for key in source_keys):
        raise ValueError("分析通道来源已经失效，请重新选择。")


def analyze_record(
    record: ImageRecord,
    line_width_um: float,
    smoothing_sigma: float,
    background_percentile: float,
    compute_sd: bool = False,
) -> None:
    if not record.loaded:
        load_record_images(record)
    if record.roi is None:
        raise ValueError("请先生成并调整 ROI。")
    if record.line is None:
        raise ValueError("请先在 ROI 内画扫描线。")
    validate_analysis_channels(record)
    x0, y0, width, height = record.roi
    x1, y1, x2, y2 = record.line
    if math.hypot(x2 - x1, y2 - y1) < 2:
        raise ValueError("扫描线太短，请重新绘制。")
    line_width_px = max(1, int(round(line_width_um / record.pixel_size_um)))
    profile_length = 0
    for analysis_channel in record.analysis_channels:
        source = record.source_channel(analysis_channel.source_key)
        assert source is not None
        samples = profile_line(
            source.data,
            (y1, x1),
            (y2, x2),
            linewidth=line_width_px,
            mode="constant",
            cval=0,
            reduce_func=None,
        ).astype(float)
        samples = np.atleast_2d(samples.T).T
        values = samples.mean(axis=1)
        deviations = (
            samples.std(axis=1, ddof=1)
            if compute_sd and samples.shape[1] > 1
            else None
        )
        if smoothing_sigma > 0:
            values = gaussian_filter1d(values, sigma=smoothing_sigma)
            if deviations is not None:
                deviations = gaussian_filter1d(deviations, sigma=smoothing_sigma)
        if background_percentile > 0:
            crop = source.data[y0 : y0 + height, x0 : x0 + width]
            values = np.clip(
                values - np.percentile(crop, background_percentile), 0, None
            )
        analysis_channel.profile = values
        analysis_channel.profile_sd = deviations
        profile_length = len(values)
    record.distance_um = np.linspace(
        0,
        math.hypot(x2 - x1, y2 - y1) * record.pixel_size_um,
        profile_length,
    )
    record.line_width_px = line_width_px
    record.analysis_line_width_um = line_width_um
    record.analysis_smoothing_sigma = smoothing_sigma
    record.analysis_background_percentile = background_percentile
    record.analysis_compute_sd = compute_sd
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
    display = composite_from_channels(
        record, {channel.key for channel in record.source_channels}
    )
    image = Image.fromarray(display, mode="RGB")
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
    assert record.roi is not None
    x, y, width, height = record.roi
    full_composite = composite_from_channels(
        record, {channel.key for channel in record.source_channels}
    )
    composite = Image.fromarray(
        full_composite[y : y + height, x : x + width], mode="RGB"
    )
    images = {"composite": composite}
    for index, analysis_channel in enumerate(record.analysis_channels, start=1):
        source = record.source_channel(analysis_channel.source_key)
        assert source is not None
        images[f"channel_{index}"] = colorize_channel(
            source.data[y : y + height, x : x + width],
            hex_to_rgb(analysis_channel.color),
        )
    return images


def make_channel_panel(
    record: ImageRecord,
    roi_images: dict[str, Image.Image],
) -> Figure:
    entries = [("composite", "Composite", "black")]
    entries.extend(
        (f"channel_{index}", channel.name, channel.color)
        for index, channel in enumerate(record.analysis_channels, start=1)
    )
    columns = min(3, max(1, math.ceil(math.sqrt(len(entries)))))
    rows = math.ceil(len(entries) / columns)
    panel = Figure(
        figsize=(columns * 3.5, rows * 3.2),
        facecolor="white",
        constrained_layout=True,
    )
    for index, (key, title, title_color) in enumerate(entries, start=1):
        axis = panel.add_subplot(rows, columns, index)
        axis.imshow(add_roi_scan_line(roi_images[key], record))
        axis.set_axis_off()
        axis.set_title(title, color=title_color, fontsize=11, fontweight="bold")
    panel.suptitle(f"{record.name} — ROI channel view", fontsize=12)
    return panel


def plot_profile_curves(axis, record: ImageRecord, lw: float = 1.6) -> None:
    assert record.distance_um is not None
    for channel in record.analysis_channels:
        assert channel.profile is not None
        if channel.profile_sd is not None:
            axis.fill_between(
                record.distance_um,
                np.clip(channel.profile - channel.profile_sd, 0, None),
                channel.profile + channel.profile_sd,
                color=channel.color,
                alpha=0.18,
                lw=0,
            )
        axis.plot(
            record.distance_um,
            channel.profile,
            color=channel.color,
            lw=lw,
            label=channel.name,
        )


def make_curve_figure(record: ImageRecord) -> Figure:
    assert record.distance_um is not None
    figure = Figure(figsize=(5.2, 3.5), facecolor="white", constrained_layout=True)
    axis = figure.add_subplot(111)
    plot_profile_curves(axis, record, lw=1.6)
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
    validate_analysis_channels(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(record.path)
    csv_path = output_dir / f"{stem}_profile.csv"
    overlay_path = output_dir / f"{stem}_overlay.png"
    roi_composite_path = output_dir / f"{stem}_ROI_composite.png"
    roi_overlay_path = output_dir / f"{stem}_ROI_composite_overlay.png"
    roi_channel_paths = [
        output_dir / f"{stem}_ROI_{safe_label(channel.name)}.png"
        for channel in record.analysis_channels
    ]
    channel_panel_png = output_dir / f"{stem}_ROI_channels_panel.png"
    channel_panel_pdf = output_dir / f"{stem}_ROI_channels_panel.pdf"
    curve_png = output_dir / f"{stem}_curve.png"
    curve_pdf = output_dir / f"{stem}_curve.pdf"
    panel_png = output_dir / f"{stem}_analysis_panel.png"
    metadata_path = output_dir / f"{stem}_analysis.json"

    assert record.distance_um is not None
    header = ["distance_um"]
    columns: list[np.ndarray] = [record.distance_um]
    for channel in record.analysis_channels:
        header.append(f"{channel.name}_AU")
        columns.append(channel.profile)
        if channel.profile_sd is not None:
            header.append(f"{channel.name}_SD")
            columns.append(channel.profile_sd)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*columns))

    overlay = make_overlay(record)
    overlay.save(overlay_path, dpi=(300, 300))
    roi_images = make_roi_channel_images(record)
    roi_images["composite"].save(roi_composite_path, dpi=(300, 300))
    for index, path in enumerate(roi_channel_paths, start=1):
        roi_images[f"channel_{index}"].save(path, dpi=(300, 300))
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
    plot_profile_curves(curve_axis, record, lw=1.5)
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
        "sd_enabled": record.analysis_compute_sd,
        "sd_definition": (
            "per-point sample SD (ddof=1) across the sampling strip width, "
            "smoothed with the same sigma as the mean"
            if record.analysis_compute_sd
            else None
        ),
        "coordinate_system": "full-resolution image pixels",
        "analysis_channels": [
            {
                "name": channel.name,
                "source_key": channel.source_key,
                "plot_color": channel.color,
                "source_file": str(source.source_path) if source and source.source_path else None,
            }
            for channel in record.analysis_channels
            for source in [record.source_channel(channel.source_key)]
        ],
        "displayed_source_channels": sorted(record.display_source_keys),
        "note": "Exported pseudocolored TIFF intensities are descriptive A.U.; use raw CZI for rigorous between-sample quantification.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    exported = {
        "csv": csv_path,
        "full_overlay": overlay_path,
        "roi_composite": roi_composite_path,
        "roi_overlay": roi_overlay_path,
        "channel_panel_png": channel_panel_png,
        "channel_panel_pdf": channel_panel_pdf,
        "curve_png": curve_png,
        "curve_pdf": curve_pdf,
        "panel": panel_png,
        "metadata": metadata_path,
    }
    exported.update(
        {f"roi_channel_{index}": path for index, path in enumerate(roi_channel_paths, start=1)}
    )
    return exported


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
        self.visible_source_keys: set[str] | None = None
        self.selector: RectangleSelector | None = None
        self.line_connection: int | None = None
        self.line_points: list[tuple[float, float]] = []
        self.mode = "idle"
        self.roi_snapshot: tuple | None = None
        self.view: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._pan_start: tuple | None = None
        self.mpl_connect("key_press_event", self._on_key)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_pan_press)
        self.mpl_connect("motion_notify_event", self._on_pan_motion)
        self.mpl_connect("button_release_event", self._on_pan_release)
        self.render()

    def set_record(self, record: ImageRecord | None) -> None:
        self.cancel_mode()
        if record is not self.record:
            self.view = None
        self.record = record
        self.visible_source_keys = set(record.display_source_keys) if record else None
        self.render()

    def set_visible_sources(self, source_keys: set[str]) -> None:
        if self.mode == "roi":
            self.finish_roi_edit(commit=True)
        elif self.mode == "line":
            self.cancel_mode()
        self.visible_source_keys = set(source_keys)
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
        display = composite_from_channels(self.record, self.visible_source_keys)
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
        if self.view is not None:
            self.axis.set_xlim(self.view[0])
            self.axis.set_ylim(self.view[1])
        self.draw_idle()

    def _full_extent(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if self.record is None or self.record.rgb is None:
            return None
        image_height, image_width = self.record.rgb.shape[:2]
        return (-0.5, image_width - 0.5), (image_height - 0.5, -0.5)

    def _apply_view(
        self, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> None:
        extent = self._full_extent()
        if extent is None:
            return
        (full_x0, full_x1), (full_y0, full_y1) = extent
        span_x = min(xlim[1] - xlim[0], full_x1 - full_x0)
        span_y = min(ylim[0] - ylim[1], full_y0 - full_y1)
        x0 = float(np.clip(xlim[0], full_x0, full_x1 - span_x))
        y1 = float(np.clip(ylim[1], full_y1, full_y0 - span_y))
        if span_x >= full_x1 - full_x0 and span_y >= full_y0 - full_y1:
            self.view = None
            self.axis.set_xlim(full_x0, full_x1)
            self.axis.set_ylim(full_y0, full_y1)
        else:
            self.view = ((x0, x0 + span_x), (y1 + span_y, y1))
            self.axis.set_xlim(self.view[0])
            self.axis.set_ylim(self.view[1])
        self.draw_idle()

    def reset_view(self) -> None:
        if self.view is None:
            return
        self.view = None
        extent = self._full_extent()
        if extent is not None:
            self.axis.set_xlim(extent[0])
            self.axis.set_ylim(extent[1])
            self.draw_idle()

    def _on_scroll(self, event) -> None:
        if (
            self.record is None
            or not self.record.loaded
            or event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        factor = 1 / 1.3 if event.button == "up" else 1.3
        x0, x1 = self.axis.get_xlim()
        y0, y1 = self.axis.get_ylim()
        if factor < 1 and min(x1 - x0, y0 - y1) * factor < 20:
            return
        cx, cy = float(event.xdata), float(event.ydata)
        self._apply_view(
            (cx - (cx - x0) * factor, cx + (x1 - cx) * factor),
            (cy + (y0 - cy) * factor, cy - (cy - y1) * factor),
        )

    def _on_pan_press(self, event) -> None:
        if self.record is None or not self.record.loaded or event.inaxes is not self.axis:
            return
        if event.button == 1 and self.mode == "idle" and event.dblclick:
            self.reset_view()
            return
        pannable = event.button == 2 or (event.button == 1 and self.mode == "idle")
        if pannable and self.view is not None:
            self._pan_start = (
                event.x,
                event.y,
                self.axis.get_xlim(),
                self.axis.get_ylim(),
            )
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_pan_motion(self, event) -> None:
        if self._pan_start is None:
            return
        press_x, press_y, (x0, x1), (y0, y1) = self._pan_start
        bbox = self.axis.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx = (event.x - press_x) * (x1 - x0) / bbox.width
        dy = (event.y - press_y) * (y0 - y1) / bbox.height
        self._apply_view((x0 - dx, x1 - dx), (y0 + dy, y1 + dy))

    def _on_pan_release(self, event) -> None:
        if self._pan_start is not None:
            self._pan_start = None
            self.unsetCursor()

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
            [channel.profile for channel in self.record.analysis_channels],
            self.record.line_width_px,
            self.record.dirty,
        )
        self.record.line = None
        self.record.invalidate_analysis()
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
        self.record.invalidate_analysis()
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
                profiles,
                record.line_width_px,
                record.dirty,
            ) = snapshot
            for channel, profile in zip(record.analysis_channels, profiles):
                channel.profile = profile
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
        self.record.invalidate_analysis()
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
            self.record.invalidate_analysis()
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
        self.axis.clear()
        plot_profile_curves(self.axis, record, lw=1.5)
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
        self.analysis_row_widgets: list[dict[str, QWidget]] = []
        self.display_channel_checks: dict[str, QCheckBox] = {}
        self.setWindowTitle(APP_TITLE)
        self.resize(1550, 920)
        self.setMinimumSize(920, 620)
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
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.main_splitter)

        left_panel = QFrame()
        left_panel.setObjectName("sidePanel")
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(330)
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
        self.main_splitter.addWidget(left_panel)

        center_panel = QWidget()
        center_panel.setMinimumWidth(330)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(6, 0, 6, 0)
        image_header = QHBoxLayout()
        image_header.addWidget(QLabel("图像与 Overlay"))
        self.display_channel_widget = QWidget()
        self.display_channel_layout = QHBoxLayout(self.display_channel_widget)
        self.display_channel_layout.setContentsMargins(4, 0, 4, 0)
        self.display_channel_layout.setSpacing(10)
        self.display_channel_layout.addStretch(1)
        display_scroll = QScrollArea()
        display_scroll.setWidgetResizable(True)
        display_scroll.setWidget(self.display_channel_widget)
        display_scroll.setFrameShape(QFrame.Shape.NoFrame)
        display_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        display_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        display_scroll.setFixedHeight(34)
        image_header.addWidget(display_scroll, 1)
        center_layout.addLayout(image_header)
        self.image_canvas = ImageCanvas()
        self.image_toolbar = NavigationToolbar2QT(self.image_canvas, center_panel)
        self.image_toolbar.setIconSize(self.image_toolbar.iconSize())
        center_layout.addWidget(self.image_toolbar)
        center_layout.addWidget(self.image_canvas, 1)
        self.canvas_hint = QLabel("提示：滚轮缩放；工具栏可平移/复位；ROI 为黄色，扫描线为白色虚线。")
        self.canvas_hint.setObjectName("hint")
        center_layout.addWidget(self.canvas_hint)
        self.main_splitter.addWidget(center_panel)

        right_panel = QFrame()
        right_panel.setObjectName("sidePanel")
        right_panel.setMinimumWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_scroll.setMinimumWidth(330)
        self.right_scroll.setMaximumWidth(540)
        self.right_scroll.setWidget(right_panel)

        parameters = QGroupBox("分析参数")
        parameter_form = QFormLayout(parameters)
        parameter_form.setContentsMargins(8, 8, 8, 6)
        parameter_form.setVerticalSpacing(3)
        parameter_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
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
        self.sd_check = QCheckBox("SD 误差带（均值 ± SD）")
        self.sd_check.setToolTip(
            "沿扫描带宽方向计算每个采样点的标准差；\n曲线显示半透明误差带，CSV 增加 *_SD 列。\n带宽为 1 px 时无法计算。"
        )
        parameter_form.addRow("像素尺寸", self.pixel_spin)
        parameter_form.addRow("扫描带宽", self.width_spin)
        parameter_form.addRow("平滑 σ", self.smooth_spin)
        parameter_form.addRow("背景分位数", self.background_spin)
        parameter_form.addRow(self.sd_check)
        right_layout.addWidget(parameters)

        analysis_group = QGroupBox("分析通道")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setContentsMargins(8, 8, 8, 7)
        analysis_layout.setSpacing(5)
        analysis_hint = QLabel("点击添加后，在弹窗中设置来源、名称和颜色。")
        analysis_hint.setObjectName("hint")
        analysis_hint.setWordWrap(True)
        analysis_layout.addWidget(analysis_hint)
        self.analysis_rows_widget = QWidget()
        self.analysis_rows_layout = QVBoxLayout(self.analysis_rows_widget)
        self.analysis_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_rows_layout.setSpacing(4)
        analysis_layout.addWidget(self.analysis_rows_widget)
        self.add_analysis_channel_button = QPushButton("＋ 添加分析通道")
        analysis_layout.addWidget(self.add_analysis_channel_button)
        right_layout.addWidget(analysis_group)

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
        self.roi_button = QPushButton("① 按输入尺寸生成可调 ROI 框")
        self.roi_button.setToolTip("生成后可直接拖动框体移动，或拖动边/角控制点缩放")
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
            "导出：全图定位、ROI 合并图、每个分析通道、动态通道面板、曲线、CSV 和参数 JSON"
        )
        self.export_note.setObjectName("hint")
        self.export_note.setWordWrap(True)
        export_layout.addWidget(self.export_note)
        right_layout.addWidget(export_group)
        right_layout.addStretch(1)
        self.main_splitter.addWidget(self.right_scroll)
        self.main_splitter.setSizes([260, 850, 420])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("准备就绪。请导入图像。")

        self.open_button.clicked.connect(self.choose_images)
        self.folder_button.clicked.connect(self.choose_folder)
        self.image_list.currentRowChanged.connect(self.select_record)
        self.remove_button.clicked.connect(self.remove_current)
        self.clear_button.clicked.connect(self.clear_records)
        self.pixel_spin.valueChanged.connect(self.pixel_size_changed)
        self.width_spin.valueChanged.connect(self.parameters_changed)
        self.smooth_spin.valueChanged.connect(self.parameters_changed)
        self.background_spin.valueChanged.connect(self.parameters_changed)
        self.sd_check.toggled.connect(self.parameters_changed)
        self.add_analysis_channel_button.clicked.connect(self.add_analysis_channel)
        self.roi_button.clicked.connect(self.create_or_edit_roi)
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
            QFrame#channelRow { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; }
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
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
            QDoubleSpinBox, QLineEdit, QComboBox { color: #111827; background: #FFFFFF; selection-background-color: #0F766E; selection-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 3px; padding: 2px 5px; min-height: 26px; }
            QDoubleSpinBox:disabled, QLineEdit:disabled, QComboBox:disabled { color: #9CA3AF; background: #F3F4F6; }
            QComboBox::drop-down { border: none; width: 20px; }
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
            self.roi_width_spin,
            self.roi_height_spin,
            self.roi_button,
        ):
            widget.setEnabled(has_record)
        used_sources = {channel.source_key for channel in record.analysis_channels} if record else set()
        self.add_analysis_channel_button.setEnabled(
            bool(
                record
                and any(source.key not in used_sources for source in record.source_channels)
            )
        )
        for row in self.analysis_row_widgets:
            for widget in row.values():
                widget.setEnabled(has_record)
        self.line_button.setEnabled(has_roi)
        self.analyze_button.setEnabled(bool(has_line and record and record.analysis_channels))
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
            record = ImageRecord(path=path, name=display_name(path))
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
            self.rebuild_display_channel_checks()
            self.rebuild_analysis_channel_rows()
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
            self.rebuild_display_channel_checks()
            self.rebuild_analysis_channel_rows()
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

    def _clear_layout(self, layout: QVBoxLayout | QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def rebuild_display_channel_checks(self) -> None:
        self._clear_layout(self.display_channel_layout)
        self.display_channel_checks.clear()
        record = self.current_record
        if record is None or not record.loaded:
            placeholder = QLabel("导入图片后显示可用通道")
            placeholder.setObjectName("hint")
            self.display_channel_layout.addWidget(placeholder)
        else:
            for source in record.source_channels:
                checkbox = QCheckBox(source.label)
                checkbox.setChecked(source.key in record.display_source_keys)
                checkbox.setStyleSheet(f"QCheckBox {{ color: {source.color}; font-weight: 600; }}")
                checkbox.toggled.connect(self.update_channels)
                self.display_channel_checks[source.key] = checkbox
                self.display_channel_layout.addWidget(checkbox)
        self.display_channel_layout.addStretch(1)

    def update_channels(self) -> None:
        record = self.current_record
        if record is None:
            return
        selected = {
            key for key, checkbox in self.display_channel_checks.items() if checkbox.isChecked()
        }
        record.display_source_keys = selected
        self.image_canvas.set_visible_sources(selected)

    def rebuild_analysis_channel_rows(self) -> None:
        self._clear_layout(self.analysis_rows_layout)
        self.analysis_row_widgets.clear()
        record = self.current_record
        if record is None or not record.loaded:
            empty = QLabel("请先导入并选择图像。")
            empty.setObjectName("hint")
            self.analysis_rows_layout.addWidget(empty)
            return
        if not record.analysis_channels:
            empty = QLabel("尚未添加分析通道。")
            empty.setObjectName("hint")
            self.analysis_rows_layout.addWidget(empty)
            return
        for index, analysis_channel in enumerate(record.analysis_channels):
            row_widget = QFrame()
            row_widget.setObjectName("channelRow")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(7, 4, 5, 4)
            row_layout.setSpacing(6)
            color_swatch = QLabel()
            color_swatch.setFixedSize(16, 16)
            color_swatch.setStyleSheet(
                f"background: {analysis_channel.color}; border: 1px solid #64748B; border-radius: 8px;"
            )
            name_label = QLabel(analysis_channel.name)
            name_label.setStyleSheet("font-weight: 700; color: #111827;")
            source = record.source_channel(analysis_channel.source_key)
            source_label = QLabel(source.label if source else analysis_channel.source_key)
            source_label.setObjectName("hint")
            edit_button = QPushButton("编辑")
            edit_button.setMaximumWidth(52)
            remove_button = QPushButton("删除")
            remove_button.setMaximumWidth(52)
            row_layout.addWidget(color_swatch)
            row_layout.addWidget(name_label)
            row_layout.addWidget(source_label)
            row_layout.addStretch(1)
            row_layout.addWidget(edit_button)
            row_layout.addWidget(remove_button)
            self.analysis_rows_layout.addWidget(row_widget)
            self.analysis_row_widgets.append(
                {
                    "row": row_widget,
                    "color": color_swatch,
                    "name": name_label,
                    "source": source_label,
                    "edit": edit_button,
                    "remove": remove_button,
                }
            )
            edit_button.clicked.connect(
                lambda _checked=False, row_index=index: self.edit_analysis_channel(row_index)
            )
            remove_button.clicked.connect(
                lambda _checked=False, row_index=index: self.remove_analysis_channel(row_index)
            )

    def append_analysis_channel(self, source_key: str, name: str, color: str) -> None:
        record = self.current_record
        if record is None or record.source_channel(source_key) is None:
            return
        if any(channel.source_key == source_key for channel in record.analysis_channels):
            raise ValueError("同一个图像通道只能添加一次。")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("请输入通道名称。")
        record.analysis_channels.append(
            AnalysisChannel(source_key=source_key, name=clean_name, color=color.upper())
        )
        source = record.source_channel(source_key)
        if source is not None:
            source.color = color.upper()
        record.invalidate_analysis()
        self.rebuild_analysis_channel_rows()
        self.rebuild_display_channel_checks()
        self.image_canvas.render()
        self.curve_canvas.clear_plot()
        self.update_file_info()
        self._refresh_list_item(self.image_list.currentRow())
        self._update_controls()

    def add_analysis_channel(self, checked: bool = False) -> None:
        del checked
        record = self.current_record
        if record is None or not record.loaded:
            return
        used = {channel.source_key for channel in record.analysis_channels}
        source = next((item for item in record.source_channels if item.key not in used), None)
        if source is None:
            self.statusBar().showMessage("所有可用图像通道都已经加入分析。")
            return
        dialog = ChannelEditorDialog(
            record.source_channels,
            used_source_keys=used,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.append_analysis_channel(*dialog.values())

    def edit_analysis_channel(self, index: int) -> None:
        record = self.current_record
        if record is None or not (0 <= index < len(record.analysis_channels)):
            return
        channel = record.analysis_channels[index]
        used = {
            other.source_key
            for other_index, other in enumerate(record.analysis_channels)
            if other_index != index
        }
        dialog = ChannelEditorDialog(
            record.source_channels,
            used_source_keys=used,
            existing=channel,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        source_key, name, color = dialog.values()
        source_changed = channel.source_key != source_key
        channel.source_key = source_key
        channel.name = name
        channel.color = color.upper()
        source = record.source_channel(source_key)
        if source is not None:
            source.color = color.upper()
        if source_changed:
            record.invalidate_analysis()
            self.curve_canvas.clear_plot()
        else:
            self.curve_canvas.show_record(record)
        self.rebuild_analysis_channel_rows()
        self.rebuild_display_channel_checks()
        self.image_canvas.render()
        self.update_file_info()
        self._refresh_list_item(self.image_list.currentRow())
        self._update_controls()

    def remove_analysis_channel(self, index: int) -> None:
        record = self.current_record
        if record is None or not (0 <= index < len(record.analysis_channels)):
            return
        record.analysis_channels.pop(index)
        record.invalidate_analysis()
        self.rebuild_analysis_channel_rows()
        self.curve_canvas.clear_plot()
        self.update_file_info()
        self._refresh_list_item(self.image_list.currentRow())
        self._update_controls()

    def update_file_info(self) -> None:
        record = self.current_record
        if record is None or record.rgb is None:
            self.file_info.setText("尚未导入图像")
            return
        height, width = record.rgb.shape[:2]
        channel_note = "可用通道：" + "、".join(channel.label for channel in record.source_channels)
        analysis_note = (
            "分析：" + "、".join(channel.name or "（未命名）" for channel in record.analysis_channels)
            if record.analysis_channels
            else "分析：尚未添加通道"
        )
        self.file_info.setText(
            f"{record.name}\n{width} × {height} px\n"
            f"{record.pixel_size_um:.6f} µm/px\n{record.pixel_source}\n{channel_note}\n{analysis_note}"
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

    def create_or_edit_roi(self) -> None:
        if self.image_canvas.mode == "roi":
            self.image_canvas.finish_roi_edit(commit=True)
            return
        self.apply_roi_size()
        self.image_canvas.start_roi()

    def canvas_mode_changed(self, mode: str) -> None:
        if mode == "roi":
            self.roi_button.setText("完成 ROI 编辑（Enter）")
            self.roi_button.setObjectName("primaryButton")
        else:
            self.roi_button.setText("① 按输入尺寸生成可调 ROI 框")
            self.roi_button.setObjectName("")
        self.roi_button.style().unpolish(self.roi_button)
        self.roi_button.style().polish(self.roi_button)

    def pixel_size_changed(self, value: float) -> None:
        record = self.current_record
        if record is None:
            return
        record.pixel_size_um = value
        record.pixel_source = "用户在 UI 中设置"
        record.invalidate_analysis()
        self.image_canvas.render()
        self.sync_roi_inputs()
        self.update_file_info()
        self.curve_canvas.clear_plot()
        self._refresh_list_item(self.image_list.currentRow())
        self._update_controls()

    def parameters_changed(self) -> None:
        record = self.current_record
        if record is not None and record.analyzed:
            record.invalidate_analysis()
            self.curve_canvas.clear_plot()
            self._refresh_list_item(self.image_list.currentRow())
            self.statusBar().showMessage("参数已改变，请重新点击“生成曲线”。")
            self._update_controls()

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
        record.invalidate_analysis()
        self.image_canvas.cancel_mode()
        self.image_canvas.render()
        self.selection_changed()
        self.statusBar().showMessage("ROI 和扫描线已清除。")

    def clear_line(self) -> None:
        record = self.current_record
        if record is None:
            return
        record.line = None
        record.invalidate_analysis()
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
                compute_sd=self.sd_check.isChecked(),
            )
            self.curve_canvas.show_record(record)
            self._refresh_list_item(self.image_list.currentRow())
            self._update_controls()
            message = (
                f"曲线已生成：{len(record.distance_um) if record.distance_um is not None else 0} 个采样点，"
                f"扫描带宽 {record.line_width_px} px。"
            )
            if self.sd_check.isChecked() and record.line_width_px == 1:
                message += " 带宽只有 1 px，无法计算 SD，仅输出均值。"
            self.statusBar().showMessage(message)
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
            "1. 导入图像；软件会识别 RGB、灰度或同目录的 c1/c2/c3 通道。\n"
            "2. 确认像素尺寸，添加分析通道，选择来源、输入名称并按需设置颜色。\n"
            "3. 输入 ROI 宽和高，点击“生成可调 ROI 框”。\n"
            "4. 直接拖动框体移动，拖动边/角控制点缩放，按 Enter 完成。\n"
            "5. 点击“画扫描线”，在 ROI 内点击两个端点，然后生成曲线。\n"
            "6. 导出全图定位、ROI 裁剪、每个分析通道、曲线图和 CSV。\n\n"
            "视图缩放：滚轮以鼠标位置为中心放大/缩小；空闲状态下左键拖动平移，"
            "双击复位；ROI/扫描线模式下可用中键拖动平移。\n\n"
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
    if len(record.source_channels) < 2:
        raise RuntimeError("self-test needs an image with at least two channels")
    window.append_analysis_channel(
        record.source_channels[0].key,
        "Marker A",
        record.source_channels[0].color,
    )
    window.append_analysis_channel(
        record.source_channels[1].key,
        "Nuclear stain",
        record.source_channels[1].color,
    )
    window.resize(920, 620)
    window.show()
    QApplication.processEvents()
    if min(window.main_splitter.sizes()) <= 0:
        raise RuntimeError("self-test responsive splitter collapsed a panel")
    if window.right_scroll.verticalScrollBar().maximum() <= 0:
        raise RuntimeError("self-test responsive panel did not enable scrolling")
    window.roi_width_spin.setValue(70.0)
    window.roi_height_spin.setValue(50.0)
    window.create_or_edit_roi()
    if record.roi is None:
        raise RuntimeError("self-test ROI size input failed")
    _rx, _ry, input_width_px, input_height_px = record.roi
    if abs(input_width_px * record.pixel_size_um - 70.0) > record.pixel_size_um:
        raise RuntimeError("self-test ROI width input was not applied")
    if abs(input_height_px * record.pixel_size_um - 50.0) > record.pixel_size_um:
        raise RuntimeError("self-test ROI height input was not applied")
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
    if any(channel.profile_sd is not None for channel in record.analysis_channels):
        raise RuntimeError("self-test SD computed although the option is off")
    window.sd_check.setChecked(True)
    if not window.run_analysis(silent=True):
        raise RuntimeError("self-test SD analysis failed")
    canvas = window.image_canvas
    x, y, roi_width, roi_height = record.roi
    canvas._on_scroll(
        SimpleNamespace(
            inaxes=canvas.axis,
            xdata=x + roi_width / 2,
            ydata=y + roi_height / 2,
            button="up",
        )
    )
    if canvas.view is None:
        raise RuntimeError("self-test wheel zoom did not change the view")
    canvas.render()
    if abs(canvas.axis.get_xlim()[0] - canvas.view[0][0]) > 1e-6:
        raise RuntimeError("self-test zoomed view was lost after render")
    canvas.reset_view()
    if canvas.view is not None:
        raise RuntimeError("self-test view reset failed")
    if record.line_width_px > 1 and any(
        channel.profile_sd is None for channel in record.analysis_channels
    ):
        raise RuntimeError("self-test SD profiles missing")
    with tempfile.TemporaryDirectory(prefix="linescan_analyzer_test_") as folder:
        exported = export_record(record, Path(folder))
        missing = [path for path in exported.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"self-test export missing: {missing}")
        csv_header = exported["csv"].read_text(encoding="utf-8-sig").splitlines()[0]
        metadata = json.loads(exported["metadata"].read_text(encoding="utf-8"))
        if "Marker A_AU" not in csv_header or "Nuclear stain_AU" not in csv_header:
            raise RuntimeError("self-test custom signal names missing from CSV")
        if record.line_width_px > 1 and "Marker A_SD" not in csv_header:
            raise RuntimeError("self-test SD columns missing from CSV")
        channel_metadata = metadata.get("analysis_channels", [])
        if not channel_metadata or channel_metadata[0].get("name") != "Marker A":
            raise RuntimeError("self-test custom signal names missing from metadata")
        expected_size = (record.roi[2], record.roi[3])
        for key in ("roi_composite", "roi_overlay", "roi_channel_1", "roi_channel_2"):
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
