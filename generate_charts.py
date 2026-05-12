"""
Generate 5 comparison charts from results_sklearn.json and results_mlnet.json.

Outputs:
    fig1_train.png     - training time (log Y) for all 6 algorithms
    fig2_infer.png     - inference time (linear Y) for all 6 algorithms
    fig3_accuracy.png  - accuracy, 4 classifiers, zoomed Y, annotated bars
    fig4_f1.png        - F1 weighted, 4 classifiers, zoomed Y, annotated bars
    fig5_regression.png - R^2 and RMSE side by side
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).parent
SKLEARN_JSON = ROOT / "results_sklearn.json"
MLNET_JSON = ROOT / "results_mlnet.json"

SKLEARN_COLOR = "#1565C0"
MLNET_COLOR = "#B71C1C"
BAR_WIDTH = 0.35
FIGSIZE = (11, 5)
DPI = 150

ALL_ALGORITHMS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM_Linear",
    "KMeans",
    "LinearRegression",
]
ALL_LABELS = [
    "Reg. Logistică",
    "Arbore Decizie",
    "Random Forest",
    "SVM Linear",
    "K-Means",
    "Reg. Liniară",
]
CLASSIFIER_ALGORITHMS = ["LogisticRegression", "DecisionTree", "RandomForest", "SVM_Linear"]
CLASSIFIER_LABELS = ["Reg. Logistică", "Arbore Decizie", "Random Forest", "SVM Linear"]


def _apply_rc_params() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _load_results() -> tuple[dict, dict]:
    if not SKLEARN_JSON.exists():
        raise FileNotFoundError(f"Missing {SKLEARN_JSON}. Run experiments_sklearn.py first.")
    if not MLNET_JSON.exists():
        raise FileNotFoundError(
            f"Missing {MLNET_JSON}. Run MLNetBenchmark (dotnet run -c Release) first."
        )
    with SKLEARN_JSON.open("r", encoding="utf-8-sig") as f:
        sklearn_results = json.load(f)
    with MLNET_JSON.open("r", encoding="utf-8-sig") as f:
        mlnet_results = json.load(f)
    return sklearn_results, mlnet_results


def _grouped_bar(
    ax,
    labels: list[str],
    sklearn_values: list[float],
    mlnet_values: list[float],
) -> tuple[np.ndarray, list, list]:
    x = np.arange(len(labels))
    bars_sklearn = ax.bar(
        x - BAR_WIDTH / 2,
        sklearn_values,
        BAR_WIDTH,
        color=SKLEARN_COLOR,
        label="scikit-learn (Python)",
    )
    bars_mlnet = ax.bar(
        x + BAR_WIDTH / 2,
        mlnet_values,
        BAR_WIDTH,
        color=MLNET_COLOR,
        label="ML.NET (C#)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    return x, bars_sklearn, bars_mlnet


def _annotate_bars(ax, bars, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_train_times(sklearn_results: dict, mlnet_results: dict) -> None:
    sklearn_vals = [sklearn_results[a]["train_ms"] for a in ALL_ALGORITHMS]
    mlnet_vals = [mlnet_results[a]["train_ms"] for a in ALL_ALGORITHMS]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _grouped_bar(ax, ALL_LABELS, sklearn_vals, mlnet_vals)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Timp de antrenament (ms)")
    ax.set_title("Timp de antrenament: scikit-learn vs ML.NET (scară log)")
    fig.tight_layout()
    fig.savefig(ROOT / "fig1_train.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_infer_times(sklearn_results: dict, mlnet_results: dict) -> None:
    sklearn_vals = [sklearn_results[a]["infer_ms"] for a in ALL_ALGORITHMS]
    mlnet_vals = [mlnet_results[a]["infer_ms"] for a in ALL_ALGORITHMS]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _grouped_bar(ax, ALL_LABELS, sklearn_vals, mlnet_vals)
    ax.set_ylabel("Timp de inferență (ms)")
    ax.set_title("Timp de inferență: scikit-learn vs ML.NET")
    fig.tight_layout()
    fig.savefig(ROOT / "fig2_infer.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_classifier_metric(
    sklearn_results: dict,
    mlnet_results: dict,
    metric_key: str,
    y_label: str,
    title: str,
    output_filename: str,
) -> None:
    sklearn_vals = [sklearn_results[a][metric_key] for a in CLASSIFIER_ALGORITHMS]
    mlnet_vals = [mlnet_results[a][metric_key] for a in CLASSIFIER_ALGORITHMS]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _, bars_sklearn, bars_mlnet = _grouped_bar(
        ax, CLASSIFIER_LABELS, sklearn_vals, mlnet_vals
    )

    all_vals = sklearn_vals + mlnet_vals
    min_val = min(all_vals)
    max_val = max(all_vals)
    ax.set_ylim([max(0.0, min_val - 0.02), min(1.0, max_val + 0.02)])

    _annotate_bars(ax, bars_sklearn, sklearn_vals)
    _annotate_bars(ax, bars_mlnet, mlnet_vals)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(ROOT / output_filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(sklearn_results: dict, mlnet_results: dict) -> None:
    _plot_classifier_metric(
        sklearn_results,
        mlnet_results,
        metric_key="Accuracy",
        y_label="Acuratețe",
        title="Acuratețe: scikit-learn vs ML.NET",
        output_filename="fig3_accuracy.png",
    )


def plot_f1(sklearn_results: dict, mlnet_results: dict) -> None:
    _plot_classifier_metric(
        sklearn_results,
        mlnet_results,
        metric_key="F1",
        y_label="F1-score (weighted)",
        title="F1-score: scikit-learn vs ML.NET",
        output_filename="fig4_f1.png",
    )


def plot_regression(sklearn_results: dict, mlnet_results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)

    # R^2 subplot
    r2_sklearn = sklearn_results["LinearRegression"]["R2"]
    r2_mlnet = mlnet_results["LinearRegression"]["R2"]
    ax_r2 = axes[0]
    bars_r2 = ax_r2.bar(
        [0, 1],
        [r2_sklearn, r2_mlnet],
        color=[SKLEARN_COLOR, MLNET_COLOR],
        width=0.5,
    )
    ax_r2.set_xticks([0, 1])
    ax_r2.set_xticklabels(["scikit-learn", "ML.NET"])
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("Coeficient de determinare R²")
    ax_r2.grid(axis="y", linestyle="--", alpha=0.35)
    r2_min, r2_max = min(r2_sklearn, r2_mlnet), max(r2_sklearn, r2_mlnet)
    ax_r2.set_ylim([r2_min - 0.02, r2_max + 0.02])
    _annotate_bars(ax_r2, bars_r2, [r2_sklearn, r2_mlnet])

    # RMSE subplot
    rmse_sklearn = sklearn_results["LinearRegression"]["RMSE"]
    rmse_mlnet = mlnet_results["LinearRegression"]["RMSE"]
    ax_rmse = axes[1]
    bars_rmse = ax_rmse.bar(
        [0, 1],
        [rmse_sklearn, rmse_mlnet],
        color=[SKLEARN_COLOR, MLNET_COLOR],
        width=0.5,
    )
    ax_rmse.set_xticks([0, 1])
    ax_rmse.set_xticklabels(["scikit-learn", "ML.NET"])
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Eroare pătratică medie (RMSE)")
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(ax_rmse, bars_rmse, [rmse_sklearn, rmse_mlnet])

    fig.suptitle("Regresie liniară: comparație scikit-learn vs ML.NET")
    fig.tight_layout()
    fig.savefig(ROOT / "fig5_regression.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _apply_rc_params()
    sklearn_results, mlnet_results = _load_results()

    plot_train_times(sklearn_results, mlnet_results)
    plot_infer_times(sklearn_results, mlnet_results)
    plot_accuracy(sklearn_results, mlnet_results)
    plot_f1(sklearn_results, mlnet_results)
    plot_regression(sklearn_results, mlnet_results)

    print("Charts generated:")
    for fig_name in [
        "fig1_train.png",
        "fig2_infer.png",
        "fig3_accuracy.png",
        "fig4_f1.png",
        "fig5_regression.png",
    ]:
        print(f"  - {ROOT / fig_name}")


if __name__ == "__main__":
    main()
