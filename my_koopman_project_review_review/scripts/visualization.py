#!/usr/bin/env python3
"""
Paper-style loss comparison plots with common end points and tight y-limits.

This version keeps the original smoothing logic you pasted:
  log-loss space + centred rolling median + centred rolling mean(s).

Changes only for style/usability:
  - supports --figsize WIDTH HEIGHT
  - default figure is narrower
  - Optimiser is blue, Ours is red
  - target/baseline horizontal line is orange
  - speedup arrow is orange
  - full top/right/bottom/left frame is kept
  - supports both --window-frac and fixed --window as before

Example:
  python3 visualization_original_smooth_style.py \
    --files training_progress_A.csv training_progress.csv \
    --labels Optimiser Ours \
    --reference-label Optimiser \
    --common-end \
    --end-step 14070 \
    --target-mode visible-min \
    --target-label "Baseline Best Loss" \
    --window 2501 \
    --smooth-passes 3 \
    --figsize 3.3 3.2 \
    --out-prefix loss_comparison \
    --x-min-zero
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Palette from the ukiyo-e style colour card.
BLUE = (55 / 255, 103 / 255, 149 / 255)
RED = (231 / 255, 98 / 255, 84 / 255)
ORANGE = (247 / 255, 170 / 255, 88 / 255)
YELLOW = (255 / 255, 208 / 255, 111 / 255)
CYAN = (114 / 255, 188 / 255, 213 / 255)
PALE_CYAN = (170 / 255, 220 / 255, 224 / 255)
TEAL_BLUE = (82 / 255, 143 / 255, 173 / 255)
DARK_NAVY = (30 / 255, 70 / 255, 110 / 255)

# Order matters:
# first file/label = Optimiser -> blue
# second file/label = Ours      -> red
PALETTE = [
    BLUE,
    RED,
    YELLOW,
    ORANGE,
    CYAN,
    PALE_CYAN,
    TEAL_BLUE,
    DARK_NAVY,
]

ANNOTATION_COLOR = ORANGE
TARGET_COLOR = ORANGE


def load_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        with open(path, "r") as f:
            return pd.DataFrame(json.load(f))
    raise ValueError(f"Unsupported file type: {path}. Use .csv or .json.")


def infer_label(path: str) -> str:
    stem = Path(path).stem
    if stem.startswith("training_progress_"):
        stem = stem[len("training_progress_"):]
    elif stem == "training_progress":
        stem = Path(path).parent.name or stem
    return stem.replace("_", " ").title()


def make_odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def centred_roll(arr: np.ndarray, window: int, reducer: str) -> np.ndarray:
    s = pd.Series(arr)
    roll = s.rolling(window=window, center=True, min_periods=1)
    if reducer == "median":
        return roll.median().to_numpy()
    if reducer == "mean":
        return roll.mean().to_numpy()
    raise ValueError("reducer must be 'median' or 'mean'.")


def smooth_loss(
    values,
    window: Optional[int],
    window_frac: float,
    method: str,
    smooth_passes: int,
    eps: float = 1e-12,
):
    """Smooth positive losses in log10 space and map back to linear scale."""
    y = np.asarray(values, dtype=float)
    y = np.maximum(y, eps)
    log_y = np.log10(y)

    if window is None:
        window = make_odd(max(11, round(len(y) * window_frac)))
    else:
        window = make_odd(window)

    # Make the window valid for very short arrays.
    window = min(window, make_odd(len(log_y))) if len(log_y) > 1 else 1

    z = log_y.copy()
    if method == "median":
        z = centred_roll(z, window, "median")
    elif method == "mean":
        z = centred_roll(z, window, "mean")
    elif method == "median-mean":
        # Robustly remove spikes first, then produce a visually smooth curve.
        z = centred_roll(z, window, "median")
        for _ in range(max(1, smooth_passes)):
            z = centred_roll(z, window, "mean")
    else:
        raise ValueError("--method must be 'median', 'mean', or 'median-mean'.")

    if method in {"median", "mean"}:
        for _ in range(max(0, smooth_passes - 1)):
            z = centred_roll(z, window, "mean")

    return np.power(10.0, z), window


def prepare_runs(args):
    labels = args.labels or [infer_label(p) for p in args.files]
    if len(labels) != len(args.files):
        raise ValueError("--labels must have the same length as --files.")

    runs = []
    for path, label in zip(args.files, labels):
        df = load_table(path)
        required = [args.loss_col, args.runtime_col, args.step_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        df = df[required].copy().dropna()
        df[args.loss_col] = df[args.loss_col].astype(float)
        df[args.runtime_col] = df[args.runtime_col].astype(float)
        df[args.step_col] = df[args.step_col].astype(float)
        df = df[df[args.loss_col] > 0]
        df = df.sort_values(args.step_col).reset_index(drop=True)

        smoothed, used_window = smooth_loss(
            df[args.loss_col].to_numpy(),
            window=args.window,
            window_frac=args.window_frac,
            method=args.method,
            smooth_passes=args.smooth_passes,
        )
        df["loss_raw"] = df[args.loss_col]
        df["loss_smooth"] = smoothed
        runs.append({"path": path, "label": label, "df": df, "window": used_window})
    return runs


def get_reference_run(runs, reference_label):
    if reference_label is None:
        return runs[0]
    for run in runs:
        if run["label"] == reference_label or Path(run["path"]).name == reference_label or run["path"] == reference_label:
            return run
    raise ValueError("Could not find --reference-label. Available labels: " + ", ".join(r["label"] for r in runs))


def visible_end(runs, x_col, manual_end, common_end: bool):
    max_xs = [float(run["df"][x_col].max()) for run in runs if len(run["df"]) > 0]
    if not max_xs:
        raise ValueError("No data to plot.")
    natural_end = min(max_xs) if common_end else max(max_xs)
    if manual_end is None:
        return natural_end
    return min(float(manual_end), natural_end) if common_end else float(manual_end)


def crop_run_df(run, x_col, x_end):
    df = run["df"].copy()
    df = df[df[x_col] <= x_end].copy()
    return df.sort_values(x_col).reset_index(drop=True)


def compute_target_loss(reference_df, args):
    y = reference_df["loss_smooth"].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("Reference run has no visible samples after cropping. Increase --end-step/--end-runtime.")

    if args.target_loss is not None:
        return float(args.target_loss), "manual target loss"

    if args.target_mode == "visible-min":
        return float(np.min(y)), "minimum visible smoothed loss of the reference run"

    if args.target_mode == "visible-final":
        return float(y[-1]), "final visible smoothed loss of the reference run"

    if args.target_mode == "visible-tail":
        n_tail = max(3, int(math.ceil(len(y) * args.tail_frac)))
        return float(np.median(y[-n_tail:])), f"median of the last {args.tail_frac:.0%} visible reference run"

    raise ValueError("Unknown --target-mode.")


def first_crossing(df, x_col, y_col, target):
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    if len(xs) == 0:
        return None
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    hit = np.where(ys <= target)[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(xs[0])

    # Interpolate in log-loss space to match the log y-axis.
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    ly0 = math.log10(max(y0, 1e-300))
    ly1 = math.log10(max(y1, 1e-300))
    lyt = math.log10(max(target, 1e-300))
    if ly0 == ly1:
        return float(x1)
    a = (lyt - ly0) / (ly1 - ly0)
    a = max(0.0, min(1.0, a))
    return float(x0 + a * (x1 - x0))


def interpolate_y(df, x_col, y_col, x):
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    if len(xs) == 0:
        return None
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def choose_ylim(visible_dfs, target_loss, args):
    """Choose a tight log-scale y-limit from the currently visible smoothed curves."""
    if args.y_min is not None or args.y_max is not None:
        values = []
        for df in visible_dfs:
            if len(df):
                values.append(df["loss_smooth"].to_numpy(dtype=float))
        if values:
            y = np.concatenate(values)
            y = y[np.isfinite(y) & (y > 0)]
        else:
            y = np.array([], dtype=float)
        default_min = min(float(np.nanmin(y)), target_loss) if len(y) else target_loss
        default_max = max(float(np.nanmax(y)), target_loss) if len(y) else target_loss
        return (
            args.y_min if args.y_min is not None else default_min * args.y_pad_bottom,
            args.y_max if args.y_max is not None else default_max * args.y_pad_top,
        )

    values = []
    for df in visible_dfs:
        if len(df):
            values.append(df["loss_smooth"].to_numpy(dtype=float))
    if not values:
        return None

    y = np.concatenate(values)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) == 0:
        return None

    ymin = min(float(np.nanmin(y)), target_loss)
    ymax = max(float(np.nanmax(y)), target_loss)

    lower = ymin * args.y_pad_bottom
    upper = ymax * args.y_pad_top
    if lower <= 0 or upper <= lower:
        return None
    return lower, upper


def plot_axis(runs, x_col, x_label, manual_end, reference_label, output_base, args):
    x_end = visible_end(runs, x_col, manual_end=manual_end, common_end=args.common_end)
    visible = []
    for run in runs:
        visible.append({**run, "df_visible": crop_run_df(run, x_col, x_end)})

    ref_run = get_reference_run(visible, reference_label)
    target_loss, target_desc = compute_target_loss(ref_run["df_visible"], args)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    crossing = {}
    for i, run in enumerate(visible):
        color = PALETTE[i % len(PALETTE)]
        df = run["df_visible"]
        if len(df) == 0:
            crossing[run["label"]] = None
            continue

        if args.show_raw:
            ax.plot(df[x_col], df["loss_raw"], color=color, linewidth=0.35, alpha=0.08)

        ax.plot(df[x_col], df["loss_smooth"], color=color, linewidth=args.linewidth, label=run["label"])
        crossing[run["label"]] = first_crossing(df, x_col, "loss_smooth", target_loss)

    ax.axhline(target_loss, color=TARGET_COLOR, linestyle="--", linewidth=1.4, label=args.target_label)

    for i, run in enumerate(visible):
        x_hit = crossing.get(run["label"])
        if x_hit is None:
            continue
        ax.axvline(x_hit, color=PALETTE[i % len(PALETTE)], linestyle=":", linewidth=1.8)

    ylim = choose_ylim([r["df_visible"] for r in visible], target_loss, args)
    if ylim:
        ax.set_ylim(*ylim)

    if not args.no_annotate:
        ref_x = crossing.get(ref_run["label"])
        candidates = [(label, x) for label, x in crossing.items() if label != ref_run["label"] and x is not None]
        if ref_x is not None and candidates:
            fast_label, fast_x = min(candidates, key=lambda item: item[1])
            if fast_x < ref_x:
                ymin, ymax = ax.get_ylim()
                log_min = math.log10(max(ymin, 1e-300))
                log_max = math.log10(max(ymax, 1e-300))
                x_mid = (fast_x + ref_x) / 2
                ref_df = ref_run["df_visible"]
                ref_segment = ref_df[(ref_df[x_col] >= fast_x) & (ref_df[x_col] <= ref_x)]
                segment_values = ref_segment["loss_smooth"].to_numpy(dtype=float)
                segment_values = segment_values[np.isfinite(segment_values) & (segment_values > 0)]
                edge_values = [
                    y
                    for y in (
                        interpolate_y(ref_df, x_col, "loss_smooth", fast_x),
                        interpolate_y(ref_df, x_col, "loss_smooth", ref_x),
                        interpolate_y(ref_df, x_col, "loss_smooth", x_mid),
                    )
                    if y is not None and np.isfinite(y) and y > 0
                ]
                if len(segment_values):
                    edge_values.append(float(np.max(segment_values)))
                ref_anchor = max(edge_values) if edge_values else target_loss
                anchor_log = math.log10(max(ref_anchor, 1e-300))
                arrow_log = min(log_max - 0.24 * (log_max - log_min), anchor_log + 0.03 * (log_max - log_min))
                text_log = min(log_max - 0.12 * (log_max - log_min), arrow_log + 0.03 * (log_max - log_min))
                arrow_y = 10 ** arrow_log
                text_y = 10 ** text_log

                delta = ref_x - fast_x
                pct = delta / ref_x * 100.0 if ref_x else float("nan")
                unit = "s" if x_col == args.runtime_col else "steps"
                if unit == "s":
                    text = f"Δ={delta:.1f}s\n({pct:.1f}% faster)"
                else:
                    text = f"Δ={delta:.0f} steps\n({pct:.1f}% faster)"
                ax.text(
                    x_mid,
                    text_y,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=args.annotation_fontsize,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.25", alpha=0.90),
                )
                ax.annotate(
                    "",
                    xy=(fast_x, arrow_y),
                    xytext=(ref_x, arrow_y),
                    arrowprops=dict(arrowstyle="<->", color=ANNOTATION_COLOR, lw=2.4),
                )

    ax.set_xlim(left=0 if args.x_min_zero else None, right=x_end)
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=args.label_fontsize)
    ax.set_ylabel("Train Loss (log scale)", fontsize=args.label_fontsize)
    if args.title:
        ax.set_title(args.title, fontsize=args.label_fontsize)
    ax.grid(True, which="major", color="0.72", linewidth=0.9)
    ax.grid(True, which="minor", color="0.88", linewidth=0.45, alpha=0.35)
    ax.tick_params(axis="both", labelsize=args.tick_fontsize)

    # Keep the full frame, like the original good-looking version.
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color("black")
    ax.tick_params(top=False, right=False)

    ax.legend(loc=args.legend_loc, fontsize=args.legend_fontsize, frameon=True)
    fig.tight_layout()

    png = f"{output_base}.png"
    pdf = f"{output_base}.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"crossing": crossing, "target_loss": target_loss, "target_desc": target_desc, "x_end": x_end, "png": png, "pdf": pdf}


def main():
    p = argparse.ArgumentParser(description="Draw paper-style loss-vs-runtime and loss-vs-step comparison figures.")
    p.add_argument("--files", nargs="+", required=True, help="CSV/JSON logs to compare.")
    p.add_argument("--labels", nargs="+", default=None, help="Labels in the same order as --files.")
    p.add_argument("--loss-col", default="train_loss")
    p.add_argument("--runtime-col", default="runtime_sec")
    p.add_argument("--step-col", default="global_step")
    p.add_argument("--reference-label", default=None, help="Baseline run whose visible loss defines the target. Default: first run.")

    p.add_argument("--common-end", action="store_true", help="Crop every run to the same endpoint. Default endpoint is the shortest run.")
    p.add_argument("--end-step", type=float, default=None, help="Manual endpoint for the step plot. With --common-end, it is capped at the shortest run.")
    p.add_argument("--end-runtime", type=float, default=None, help="Manual endpoint for the runtime plot. With --common-end, it is capped at the shortest run.")

    p.add_argument("--target-loss", type=float, default=None, help="Manually set target loss. Overrides --target-mode.")
    p.add_argument("--target-mode", choices=["visible-tail", "visible-final", "visible-min"], default="visible-tail",
                   help="How to choose the target loss from the visible part of the reference run.")
    p.add_argument("--tail-frac", type=float, default=0.05, help="Tail fraction used by --target-mode visible-tail.")
    p.add_argument("--target-label", default="Baseline Best Loss", help="Legend label for the horizontal target line.")

    p.add_argument("--window", type=int, default=2501, help="Centered rolling window. Larger values produce smoother curves.")
    p.add_argument("--window-frac", type=float, default=0.08, help="Used only if --window is omitted.")
    p.add_argument("--method", choices=["median", "mean", "median-mean"], default="median-mean",
                   help="Smoother in log-loss space. 'median-mean' is the smoothest and most robust default.")
    p.add_argument("--smooth-passes", type=int, default=3, help="Number of centred mean passes after the first smoothing stage.")

    p.add_argument("--show-raw", action="store_true", help="Also show faint raw loss curves.")
    p.add_argument("--linewidth", type=float, default=2.4)
    p.add_argument("--figsize", nargs=2, type=float, default=(3.3, 3.2), metavar=("WIDTH", "HEIGHT"),
                   help="Figure size in inches.")
    p.add_argument("--out-prefix", default="loss_comparison")
    p.add_argument("--title", default=None)
    p.add_argument("--legend-loc", default="lower left")
    p.add_argument("--label-fontsize", type=int, default=9)
    p.add_argument("--tick-fontsize", type=int, default=8)
    p.add_argument("--legend-fontsize", type=int, default=7)
    p.add_argument("--annotation-fontsize", type=int, default=8)
    p.add_argument("--arrow-y-frac", type=float, default=0.22)
    p.add_argument("--text-y-frac", type=float, default=0.08)
    p.add_argument("--x-min-zero", action="store_true", help="Force x-axis to start at zero.")
    p.add_argument("--y-min", type=float, default=None, help="Manual lower y-limit for both plots, e.g. 1e-2.")
    p.add_argument("--y-max", type=float, default=None, help="Manual upper y-limit for both plots, e.g. 2e-1.")
    p.add_argument("--y-pad-top", type=float, default=1.18, help="Multiplicative top padding for automatic log y-limit.")
    p.add_argument("--y-pad-bottom", type=float, default=0.72, help="Multiplicative bottom padding for automatic log y-limit.")
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if args.target_loss is None and args.target_mode == "visible-tail" and not (0 < args.tail_frac <= 1):
        raise ValueError("--tail-frac must be in (0, 1].")
    if args.smooth_passes < 0:
        raise ValueError("--smooth-passes must be non-negative.")

    runs = prepare_runs(args)
    ref = get_reference_run(runs, args.reference_label)
    ref_label = ref["label"]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    runtime_result = plot_axis(
        runs, args.runtime_col, "Runtime (seconds)", args.end_runtime, ref_label, args.out_prefix + "_runtime", args
    )
    step_result = plot_axis(
        runs, args.step_col, "Step", args.end_step, ref_label, args.out_prefix + "_step", args
    )

    print(f"Smoothing: log-space centred rolling {args.method}, smooth_passes={args.smooth_passes}")
    for r in runs:
        print(f"  {r['label']}: window={r['window']}, file={r['path']}")
    print(f"Reference run: {ref_label}")

    print("Runtime plot:")
    print(f"  x_end: {runtime_result['x_end']:.6g}")
    print(f"  target loss: {runtime_result['target_loss']:.6g} ({runtime_result['target_desc']})")
    for r in runs:
        lab = r["label"]
        rt = runtime_result["crossing"].get(lab)
        print(f"  {lab}: {'not reached' if rt is None else f'{rt:.3f} s'}")
    print(f"  saved: {runtime_result['png']}, {runtime_result['pdf']}")

    print("Step plot:")
    print(f"  x_end: {step_result['x_end']:.6g}")
    print(f"  target loss: {step_result['target_loss']:.6g} ({step_result['target_desc']})")
    for r in runs:
        lab = r["label"]
        st = step_result["crossing"].get(lab)
        print(f"  {lab}: {'not reached' if st is None else f'{st:.0f} steps'}")
    print(f"  saved: {step_result['png']}, {step_result['pdf']}")


if __name__ == "__main__":
    main()
