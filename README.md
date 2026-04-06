# SWIP Tracking Suite — Usage Guide

A three-program pipeline for automated *C. elegans* thrash-rate analysis from multi-well plate videos.

## Quick Start

```
video (.avi/.mp4)  →  PySWIP  →  outputs/<video_name>/worm_results.csv
worm_results.csv   →  PySWIPR  →  outputs/<video_name>/worm_results_4solidityEtOlTp.csv
worm_results_4solidityEtOlTp.csv  →  BATRR  →  outputs/<video_name>/thrash.csv
```

---

## Prerequisites

### Python (PySWIP)

```bash
pip install -r requirements.txt
```

Requires: Python 3.8+, OpenCV, NumPy, Pandas, SciPy, scikit-image, matplotlib.

### R (PySWIPR + BATRR)

Install these R packages:

```r
install.packages(c("tidyverse", "data.table", "ggpubr", "pracma"))
```

---

## Step 1: PySWIP — Extract Worm Angles from Video

### 1.1 Run the Tracker

Pass your video path as a command-line argument:

```bash
python PySWIP_v4.py "worm videos/your_video.avi"
```

All outputs are automatically saved to `outputs/<video_name>/` (e.g., `outputs/your_video/`).

Two windows open: **Setup** (video preview) and **Labels** (slider reference).

### 1.2 Configure the Well Grid

Use the trackbars to match your multi-well plate:

| Trackbar | What It Does |
|----------|-------------|
| **Rows / Cols** | Number of well rows and columns in the grid |
| **Radius** | Well circle radius in pixels (original video scale) |
| **Start X / Start Y** | Pixel coordinates of the top-left well centre |
| **Spacing X / Spacing Y** | Horizontal and vertical distance between well centres |
| **Adjust Mode** | `0` = click to lock/unlock wells on grid; `1` = drag individual wells |
| **Ref Frame** | Which video frame to use for the preview |
| **Threshold** | Detection sensitivity (higher = less sensitive, fewer false positives) |
| **Small Obj** | Minimum blob area in px² — filters out noise |
| **Max Area** | Maximum blob area in px² — filters out merged blobs or smears |
| **FAST MODE** | `0` = show live visualisation (slow); `1` = no display (fast) |

### 1.3 Set Up the Grid

Set **Rows** and **Cols** to match the number of wells in your multi-well plate. Then use **Radius**, **Start X/Y**, and **Spacing X/Y** to roughly position and size the grid over your wells. You can fine-tune individual circle positions later during the locking step.

### 1.4 Tune Worm Detection Parameters

Adjust **Threshold**, **Small Obj**, and **Max Area** until the green overlay accurately highlights real worms while excluding noise, debris, and bubbles. You can use the **Ref Frame** trackbar (or press **j** to jump) to skip to any frame that shows worms clearly — use whichever frame helps you dial in the best detection across all wells.

### 1.5 Lock Each Well at Its Starting Frame

The frame displayed when you click to lock a well becomes that well's **starting frame** for analysis. Follow this procedure for each well you want to analyse:

1. Set **Adjust Mode = 0** (grid mode).
2. Use the **Ref Frame** trackbar (or press **j**) to skip to the beginning frame for the well you want to lock.
3. Click the corresponding grid circle to **lock** it (it turns red). This locks in both the well and its starting frame.
4. If the circle position needs fine-tuning, set **Adjust Mode = 1** (drag mode) and drag the circle to centre it on the worm, then set **Adjust Mode = 0** again.
5. Move on to the next well and repeat from step 1.

> [!WARNING]
> Be deliberate about which frame is displayed when you lock each well — that frame becomes the well's starting point for all analysis. Locking a well at the wrong frame will produce inaccurate results.

### 1.6 Start Analysis

When all wells are locked and positioned, press **SPACE** to save the config and begin frame-by-frame analysis.

| Key | Action |
|-----|--------|
| **q** | Quit and save results to `worm_results.csv` |

During analysis (FAST MODE = 0), two additional windows appear:
- **Live Dashboard** — shows per-well status and active objects
- **Slow Analysis** — live video overlay with spine and contour visualisation

### 1.6 Output: `worm_results.csv`

One row per detected object per frame, with 30 columns. All outputs are saved to `outputs/<video_name>/`.

| Group | Key Columns |
|-------|------------|
| Identity | `Rel_Frame`, `TimeID`, `Well`, `ObjectID` |
| Body geometry | `Angle` (bending angle, degrees), `Area`, `Net_Spine_Len`, `Spine_Arc_Len`, width stats |
| Contour/boundary | `Perimeter`, `Circularity`, `Solidity`, `Eccentricity`, tip angles |
| Position | `Centroid_X/Y`, `Head_X/Y`, `Tail_X/Y` |

**Time windows:** By default, PySWIP analyses 1-minute windows every 10 minutes (frames 0–1800, 18000–19800, … at 30 fps). Each window gets a `TimeID` (1, 2, 3, …).

---

## Step 2: PySWIPR — Clean the Data

### 2.1 Run the Filtering Script

```bash
Rscript PySWIPR_v1.R outputs/my_video/
```

The script reads `worm_results.csv` from the target directory and applies four sequential filters.

### 2.2 The Four Filters

| Stage | What It Removes | Output File |
|-------|----------------|-------------|
| **1 — Solidity** | Rows with `Solidity > 1` (numerically impossible) | `worm_results_1solidity.csv` |
| **2 — Existence Time** | Objects appearing in fewer frames than the first quartile (bottom 25% — almost always noise) | `worm_results_2solidityEt.csv` |
| **3 — Overlap** | When two objects coexist in the same well/time, keeps the one with higher angle standard deviation (the real worm) | `worm_results_3solidityEtOl.csv` |
| **4 — Time Point** | (Well, TimeID) groups with fewer than 900 valid rows (i.e., >50% data missing) | `worm_results_4solidityEtOlTp.csv` |

### 2.3 Inspect Intermediate Files

Each checkpoint CSV lets you verify how much data was removed at each stage. Compare row counts between stages to ensure filtering is not overly aggressive.

### 2.4 Output: `worm_results_4solidityEtOlTp.csv`

Same column structure as the raw CSV, but with debris, short-lived objects, overlapping duplicates, and data-sparse windows removed. This is the input to BATRR.

---

## Step 3: BATRR — Calculate Thrash Rates

### 3.1 Run the Thrash Calculation

```bash
Rscript BATRR_v1.R outputs/my_video/
```

### 3.2 How It Works

1. **Angular velocity** — computes frame-to-frame change in `Angle`
2. **Peak detection** — counts peaks and troughs in the velocity signal using `pracma::findpeaks`
   - `minpeakheight = 20` deg/frame — excludes tiny wobbles of paralysed worms
   - `minpeakdistance = 8` frames — prevents double-counting noisy spikes
3. **Thrash count** — `(peaks + troughs) / 2`, rounded down
4. **Normalisation** — converts raw count to thrashes per minute (TPM)

### 3.3 Output: `thrash.csv`

| Column | Description |
|--------|------------|
| `TimeID` | Which 10-minute window |
| `Well` | Which well (e.g., R0C1) |
| `Frames` | Number of valid data points used |
| `Thrashes` | Raw thrash count |
| `Thrash rate` | Thrashes per minute (TPM) — the final result |

Paralysed wells (no detectable movement) receive a thrash rate of **0**.

---

## Tips and Troubleshooting

### Lighting and Video Quality

- Use even, diffuse illumination. Shadows and reflections cause false detections.
- `.avi` format is preferred; `.mp4` works but may have codec issues with OpenCV.
- Record at 30 fps. The thrash-counting parameters are calibrated for this frame rate.

### Threshold Tuning

- **Too many detections** (noise, bubbles): increase **Threshold** and **Small Obj**.
- **Worm not fully detected**: decrease **Threshold** or increase **Max Area**.
- Always verify with the green-highlighted preview before starting analysis.

### Paralysed Worms

PySWIP is designed to detect paralysed worms from frame 0. The background model uses a blurred initialisation that does not absorb stationary objects. If a paralysed worm is not detected:
- Check that **Threshold** is not set too high
- Verify the well is correctly positioned over the worm

### Multiple Objects per Well

PySWIP tracks all blobs independently. PySWIPR resolves overlaps by keeping the object with the highest angle variability (the real worm). If you genuinely have multiple worms per well, the pipeline is not designed for that use case.

### Adjusting Time Windows

To change the analysis windows, edit `TARGET_WINDOWS` in `PySWIP_v4.py` (line 60):

```python
FPS = 30
# Default: 1-minute windows every 10 minutes, for 90 minutes
TARGET_WINDOWS = [(int(i * 60 * FPS), int((i + 1) * 60 * FPS)) for i in range(0, 91, 10)]
```

Each tuple is `(start_frame, end_frame)`. Adjust the range and step to match your experimental design.

### Config File

When you press SPACE in Setup mode, all slider values are saved to `setup_config.json` inside `outputs/<video_name>/`. On subsequent runs of the same video, these values load automatically. To reset, delete the file or edit it manually.

> [!NOTE]
> For technical details on how this algorithm works, check out the technicalities_accessible.pdf within the `documentation` folder.

