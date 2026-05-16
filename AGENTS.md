# AGENTS.md — SWIP Automation Suite

## What This Is

A three-step pipeline for automated _C. elegans_ thrash-rate analysis from multi-well plate videos:

```
video (.avi/.mp4)  →  PySWIP_v4.py  →  outputs/<video_name>/worm_results.csv
worm_results.csv   →  PySWIPR_v1.R  →  outputs/<video_name>/worm_results_4solidityEtOlTp.csv
worm_results_4solidityEtOlTp.csv  →  BATRR_v1.R  →  outputs/<video_name>/thrash.csv
```

## Developer Commands

```bash
# Python env (already created)
source swipautomationsuite/bin/activate

# Install Python deps
pip install -r requirements.txt

# Step 1: Run tracker (interactive GUI — requires display)
python PySWIP_v4.py "path/to/video.avi"

# Step 2: Clean data
Rscript PySWIPR_v1.R outputs/my_video/

# Step 3: Calculate thrash rates
Rscript BATRR_v1.R outputs/my_video/
```

## Architecture

- **PySWIP_v4.py** — Python. Interactive GUI for well-grid setup, then frame-by-frame worm tracking. Outputs go to `outputs/<video_name>/`. Config saved to `outputs/<video_name>/setup_config.json`.
- **PySWIPR_v1.R** — R. Four sequential debris filters. Each writes a checkpoint CSV. Final output: `worm_results_4solidityEtOlTp.csv`.
- **BATRR_v1.R** — R. Peak detection on angular velocity → thrashes per minute. Output: `thrash.csv`.

## Critical Gotchas

- **Video path is a CLI argument**. Exactly one required: `python PySWIP_v4.py "worm videos/test.avi"`. Exits with usage help if missing or extra args given.
- **R scripts require a directory argument**. Exactly one required: `Rscript PySWIPR_v1.R outputs/my_video/`. Exits with usage help if missing.
- **FPS = 30** is assumed throughout. Thrash normalisation uses `Frames * 1800 / 60` which presumes 30 fps and 1-minute windows.
- **Default time windows**: 1-minute windows every 10 minutes, from frame 0 to 90 min. Edit `TARGET_WINDOWS` at `PySWIP_v4.py:60` to change.
- **No tests, no CI, no linting**. Verification is manual: inspect checkpoint CSV row counts and final `thrash.csv`.

## Pending Work (from TODOs)

- ~~Reorganize so videos go in `worm_videos/`, outputs go in `outputs/<video_name>/`~~ — **Done**. Videos are passed via CLI arg; outputs auto-organize into `outputs/<video_name>/`.

## R Dependencies

`tidyverse`, `data.table`, `ggpubr`, `pracma`
