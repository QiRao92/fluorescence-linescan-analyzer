# Fluorescence Line-scan Analyzer

A desktop GUI for drawing and editing microscopy ROIs, extracting two-channel fluorescence line profiles, and exporting publication-ready ROI/channel panels.

![Application interface](gui_interface_preview_v2.png)

## Features

- Import individual images or recursively load a folder.
- Automatically detect ZEISS-exported `c1/c2/c3` sibling TIFF channels.
- Draw, move, and resize an interactive rectangular ROI.
- Enter exact ROI width and height in micrometres.
- Draw a line scan by selecting two endpoints inside the ROI.
- Set custom names for the two analyzed signals.
- Configure pixel calibration, sampling-strip width, smoothing, and background subtraction.
- Preview the two intensity profiles directly in the application.
- Export full-image QC overlays, cropped ROI images, individual channel images, a four-panel channel view, profile CSV, curves, and JSON metadata.

## Channel convention

For ZEISS `c1-3.tif` exports, the default mapping is:

- `c1`: F-actin display channel (cyan)
- `c2`: Signal 1 / red channel
- `c3`: Signal 2 / blue channel

Signal names are editable in the GUI and are propagated to legends, CSV headers, filenames, channel panels, and metadata.

For ordinary RGB files without sibling channel TIFFs, red is used as Signal 1, blue as Signal 2, and green for the F-actin preview.

## Installation

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

On Windows, double-click `open_HNF4A_analyzer.bat`, or run:

```powershell
python hnf4a_linescan_gui.py
```

Images can also be passed on the command line:

```powershell
python hnf4a_linescan_gui.py image1.tif image2.tif
```

## Workflow

1. Import one or more composite images.
2. Verify or enter the pixel size in `µm/px`.
3. Enter the two signal names.
4. Draw/edit an ROI, or enter exact ROI width and height.
5. Select two scan-line endpoints inside the ROI.
6. Generate the profiles and inspect the result.
7. Export the current result or batch-export all analyzed images.

## Exported files

Each analyzed image can produce:

- `*_profile.csv`: distance and the two signal intensity profiles.
- `*_overlay.png`: full-image ROI and line location.
- `*_ROI_composite.png`: clean cropped composite ROI.
- `*_ROI_composite_overlay.png`: cropped composite ROI with scan line.
- `*_ROI_F-actin.png`: clean F-actin ROI.
- `*_ROI_<signal>.png`: clean Signal 1 and Signal 2 ROIs.
- `*_ROI_channels_panel.png/.pdf`: composite plus three channel views with the scan line.
- `*_curve.png/.pdf`: line-profile curves.
- `*_analysis_panel.png`: cropped ROI and curve in one panel.
- `*_analysis.json`: calibration, ROI, line, signal, and processing metadata.

## Scientific note

Line profiles extracted from 8-bit pseudocolored exports are descriptive displayed intensities (A.U.). For rigorous between-sample fluorescence quantification, use original calibrated CZI or other raw 16-bit data and a validated segmentation/quantification workflow.

Chinese instructions are available in [README_GUI_中文.md](README_GUI_中文.md).
