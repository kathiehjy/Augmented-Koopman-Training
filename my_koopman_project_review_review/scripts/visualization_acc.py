#!/usr/bin/env python3
"""
Paper-style acc comparison plots with common end points and tight y-limits.

This version keeps the original smoothing logic you pasted:
  log-acc space + centred rolling median + centred rolling mean(s).

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
    --target-mode visible-best \
    --target-label "Baseline Best acc" \
    --window 2501 \
    --smooth-passes 3 \
    --figsize 3.3 3.2 \
    --out-prefix acc_comparison \
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
from matplotlib.ticker import FuncFormatter

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


def optional_float(value: str):
    if value.lower() == "none":
        return None
    return float(value)


def centred_roll(arr: np.ndarray, window: int, reducer: str) -> np.ndarray:
    s = pd.Series(arr)
    roll = s.rolling(window=window, center=True, min_periods=1)
    if reducer == "median":
        return roll.median().to_numpy()
    if reducer == "mean":
        return roll.mean().to_numpy()
    raise ValueError("reducer must be 'median' or 'mean'.")


def smooth_acc(
    values,
    window: Optional[int],
    window_frac: float,
    method: str,
    smooth_passes: int,
    eps: float = 1e-12,
):
    """Smooth positive acces in log10 space and map back to linear scale."""
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
        required = [args.acc_col, args.runtime_col, args.step_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        df = df[required].copy().dropna()
        df[args.acc_col] = df[args.acc_col].astype(float)
        df[args.runtime_col] = df[args.runtime_col].astype(float)
        df[args.step_col] = df[args.step_col].astype(float)
        df = df[df[args.acc_col] > 0]
        df = df.sort_values(args.step_col).reset_index(drop=True)

        smoothed, used_window = smooth_acc(
            df[args.acc_col].to_numpy(),
            window=args.window,
            window_frac=args.window_frac,
            method=args.method,
            smooth_passes=args.smooth_passes,
        )
        df["acc_raw"] = df[args.acc_col]
        df["acc_smooth"] = smoothed
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


def compute_target_acc(reference_df, args):
    y = reference_df["acc_smooth"].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("Reference run has no visible samples after cropping. Increase --end-step/--end-runtime.")

    if args.target_acc is not None:
        return float(args.target_acc), "manual target acc"

    if args.target_mode == "visible-best":
        return float(np.max(y)), "best visible smoothed acc of the reference run"

    if args.target_mode == "visible-final":
        return float(y[-1]), "final visible smoothed acc of the reference run"

    if args.target_mode == "visible-tail":
        n_tail = max(3, int(math.ceil(len(y) * args.tail_frac)))
        return float(np.median(y[-n_tail:])), f"median of the last {args.tail_frac:.0%} visible reference run"

    raise ValueError("Unknown --target-mode.")


def best_accuracy_point(df, x_col, y_col):
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    if len(xs) == 0:
        return None

    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return None

    xs = xs[finite]
    ys = ys[finite]
    best_idx = int(np.argmax(ys))
    return float(xs[best_idx]), float(ys[best_idx])


def first_crossing(df, x_col, y_col, target):
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    if len(xs) == 0:
        return None
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    hit = np.where(ys >= target)[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(xs[0])

    # Interpolate in linear accuracy space to match the normal y-axis.
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    if y0 == y1:
        return float(x1)
    a = (target - y0) / (y1 - y0)
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


def comparison_crossing(crossing, reference_label, preferred_label="Ours"):
    preferred_x = crossing.get(preferred_label)
    if preferred_label != reference_label and preferred_x is not None:
        return preferred_label, preferred_x

    candidates = [
        (label, x)
        for label, x in crossing.items()
        if label != reference_label and x is not None
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[1])


def choose_ylim(visible_dfs, target_acc, args):
    """Choose a tight normal-scale y-limit from the currently visible smoothed curves."""
    if args.y_min is not None or args.y_max is not None:
        values = []
        for df in visible_dfs:
            if len(df):
                values.append(df["acc_smooth"].to_numpy(dtype=float))
        if values:
            y = np.concatenate(values)
            y = y[np.isfinite(y) & (y > 0)]
        else:
            y = np.array([], dtype=float)
        default_min = min(float(np.nanmin(y)), target_acc) if len(y) else target_acc
        default_max = max(float(np.nanmax(y)), target_acc) if len(y) else target_acc
        span = max(default_max - default_min, 1e-12)
        lower = args.y_min if args.y_min is not None else default_min - span * args.y_pad_bottom
        upper = args.y_max if args.y_max is not None else default_max + span * args.y_pad_top
        if args.y_max is None and args.acc_max is not None:
            upper = min(upper, args.acc_max)
        return lower, upper

    values = []
    for df in visible_dfs:
        if len(df):
            values.append(df["acc_smooth"].to_numpy(dtype=float))
    if not values:
        return None

    y = np.concatenate(values)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) == 0:
        return None

    ymin = min(float(np.nanmin(y)), target_acc)
    ymax = max(float(np.nanmax(y)), target_acc)

    span = max(ymax - ymin, 1e-12)
    lower = ymin - span * args.y_pad_bottom
    upper = ymax + span * args.y_pad_top
    if args.acc_max is not None:
        upper = min(upper, args.acc_max)
    if upper <= lower:
        return None
    return lower, upper


def plot_axis(runs, x_col, x_label, manual_end, reference_label, output_base, args):
    x_end = visible_end(runs, x_col, manual_end=manual_end, common_end=args.common_end)
    visible = []
    for run in runs:
        visible.append({**run, "df_visible": crop_run_df(run, x_col, x_end)})

    ref_run = get_reference_run(visible, reference_label)
    target_acc, target_desc = compute_target_acc(ref_run["df_visible"], args)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    crossing = {}
    best_points = {}
    for i, run in enumerate(visible):
        color = PALETTE[i % len(PALETTE)]
        df = run["df_visible"]
        if len(df) == 0:
            crossing[run["label"]] = None
            best_points[run["label"]] = None
            continue

        if args.show_raw:
            ax.plot(df[x_col], df["acc_raw"], color=color, linewidth=0.35, alpha=0.08)

        ax.plot(df[x_col], df["acc_smooth"], color=color, linewidth=args.linewidth, label=run["label"])
        crossing[run["label"]] = first_crossing(df, x_col, "acc_smooth", target_acc)
        best_points[run["label"]] = best_accuracy_point(df, x_col, "acc_smooth")

    ax.axhline(target_acc, color=TARGET_COLOR, linestyle="--", linewidth=1.4, label=args.target_label)

    for i, run in enumerate(visible):
        x_hit = crossing.get(run["label"])
        if x_hit is None:
            continue
        ax.axvline(x_hit, color=PALETTE[i % len(PALETTE)], linestyle=":", linewidth=1.8)

    ylim = choose_ylim([r["df_visible"] for r in visible], target_acc, args)
    if ylim:
        ax.set_ylim(*ylim)

    if not args.no_annotate:
        ref_x = crossing.get(ref_run["label"])
        fast_label, fast_x = comparison_crossing(crossing, ref_run["label"])
        if ref_x is not None and fast_x is not None:
            if fast_x < ref_x:
                ymin, ymax = ax.get_ylim()
                y_span = max(ymax - ymin, 1e-12)
                x_mid = (fast_x + ref_x) / 2
                ref_df = ref_run["df_visible"]
                ref_segment = ref_df[(ref_df[x_col] >= fast_x) & (ref_df[x_col] <= ref_x)]
                segment_values = ref_segment["acc_smooth"].to_numpy(dtype=float)
                segment_values = segment_values[np.isfinite(segment_values)]
                edge_values = [
                    y
                    for y in (
                        interpolate_y(ref_df, x_col, "acc_smooth", fast_x),
                        interpolate_y(ref_df, x_col, "acc_smooth", ref_x),
                        interpolate_y(ref_df, x_col, "acc_smooth", x_mid),
                    )
                    if y is not None and np.isfinite(y)
                ]
                if len(segment_values):
                    edge_values.append(float(np.max(segment_values)))
                ref_anchor = max(edge_values) if edge_values else target_acc
                arrow_y = min(ymax - 0.24 * y_span, ref_anchor + 0.04 * y_span)
                text_y = min(ymax - 0.12 * y_span, arrow_y + 0.04 * y_span)

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
    ax.set_xlabel(x_label, fontsize=args.label_fontsize)
    ax.set_ylabel("Accuracy", fontsize=args.label_fontsize)
    if args.title:
        ax.set_title(args.title, fontsize=args.label_fontsize)
    ax.grid(True, which="major", color="0.72", linewidth=0.9)
    if args.hide_acc_labels_above is not None:
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, pos: "" if value > args.hide_acc_labels_above else f"{value:g}"
            )
        )
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
    return {
        "crossing": crossing,
        "best_points": best_points,
        "target_acc": target_acc,
        "target_desc": target_desc,
        "x_end": x_end,
        "png": png,
        "pdf": pdf,
    }


def main():
    p = argparse.ArgumentParser(description="Draw paper-style acc-vs-runtime and acc-vs-step comparison figures.")
    p.add_argument("--files", nargs="+", required=True, help="CSV/JSON logs to compare.")
    p.add_argument("--labels", nargs="+", default=None, help="Labels in the same order as --files.")
    p.add_argument("--acc-col", default="train_acc")
    p.add_argument("--runtime-col", default="runtime_sec")
    p.add_argument("--step-col", default="global_step")
    p.add_argument("--reference-label", default=None, help="Baseline run whose visible acc defines the target. Default: first run.")

    p.add_argument("--common-end", action="store_true", help="Crop every run to the same endpoint. Default endpoint is the shortest run.")
    p.add_argument("--end-step", type=float, default=None, help="Manual endpoint for the step plot. With --common-end, it is capped at the shortest run.")
    p.add_argument("--end-runtime", type=float, default=None, help="Manual endpoint for the runtime plot. With --common-end, it is capped at the shortest run.")

    p.add_argument("--target-acc", type=float, default=None, help="Manually set target acc. Overrides --target-mode.")
    p.add_argument("--target-mode", choices=["visible-tail", "visible-final", "visible-best"], default="visible-best",
                   help="How to choose the target acc from the visible part of the reference run.")
    p.add_argument("--tail-frac", type=float, default=0.05, help="Tail fraction used by --target-mode visible-tail.")
    p.add_argument("--target-label", default="Baseline Best acc", help="Legend label for the horizontal target line.")

    p.add_argument("--window", type=int, default=2501, help="Centered rolling window. Larger values produce smoother curves.")
    p.add_argument("--window-frac", type=float, default=0.08, help="Used only if --window is omitted.")
    p.add_argument("--method", choices=["median", "mean", "median-mean"], default="median-mean",
                   help="Smoother in log-acc space. 'median-mean' is the smoothest and most robust default.")
    p.add_argument("--smooth-passes", type=int, default=3, help="Number of centred mean passes after the first smoothing stage.")

    p.add_argument("--show-raw", action="store_true", help="Also show faint raw acc curves.")
    p.add_argument("--linewidth", type=float, default=2.4)
    p.add_argument("--figsize", nargs=2, type=float, default=(3.3, 3.2), metavar=("WIDTH", "HEIGHT"),
                   help="Figure size in inches.")
    p.add_argument("--out-prefix", default="acc_comparison")
    p.add_argument("--title", default=None)
    p.add_argument("--legend-loc", default="lower left")
    p.add_argument("--label-fontsize", type=int, default=9)
    p.add_argument("--tick-fontsize", type=int, default=8)
    p.add_argument("--legend-fontsize", type=int, default=7)
    p.add_argument("--annotation-fontsize", type=int, default=8)
    p.add_argument("--arrow-y-frac", type=float, default=0.22)
    p.add_argument("--text-y-frac", type=float, default=0.08)
    p.add_argument("--x-min-zero", action="store_true", help="Force x-axis to start at zero.")
    p.add_argument("--y-min", type=float, default=None, help="Manual lower y-limit for both plots.")
    p.add_argument("--y-max", type=float, default=None, help="Manual upper y-limit for both plots.")
    p.add_argument("--acc-max", type=optional_float, default=101.0, help="Maximum automatic y-axis value. Use --acc-max 1 for normalized accuracy or --acc-max none to disable.")
    p.add_argument("--hide-acc-labels-above", type=optional_float, default=100.0, help="Hide y-axis tick labels above this accuracy value. Use none to disable.")
    p.add_argument("--y-pad-top", type=float, default=0.18, help="Fractional top padding for automatic normal-scale y-limit.")
    p.add_argument("--y-pad-bottom", type=float, default=0.12, help="Fractional bottom padding for automatic normal-scale y-limit.")
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if args.target_acc is None and args.target_mode == "visible-tail" and not (0 < args.tail_frac <= 1):
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
    print(f"  target acc: {runtime_result['target_acc']:.6g} ({runtime_result['target_desc']})")
    for r in runs:
        lab = r["label"]
        rt = runtime_result["crossing"].get(lab)
        best = runtime_result["best_points"].get(lab)
        best_text = "best unavailable" if best is None else f"best {best[1]:.6g} at {best[0]:.3f} s"
        print(f"  {lab}: {'not reached' if rt is None else f'{rt:.3f} s'}; {best_text}")
    print(f"  saved: {runtime_result['png']}, {runtime_result['pdf']}")

    print("Step plot:")
    print(f"  x_end: {step_result['x_end']:.6g}")
    print(f"  target acc: {step_result['target_acc']:.6g} ({step_result['target_desc']})")
    for r in runs:
        lab = r["label"]
        st = step_result["crossing"].get(lab)
        best = step_result["best_points"].get(lab)
        best_text = "best unavailable" if best is None else f"best {best[1]:.6g} at {best[0]:.0f} steps"
        print(f"  {lab}: {'not reached' if st is None else f'{st:.0f} steps'}; {best_text}")
    print(f"  saved: {step_result['png']}, {step_result['pdf']}")


if __name__ == "__main__":
    main()
