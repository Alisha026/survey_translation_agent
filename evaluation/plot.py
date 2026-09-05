import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "raw": "#4C78A8",
    "corrected": "#F58518",
    "mistral": "#4C78A8",
    "qwen": "#E45756",
    "question": "#54A24B",
    "scale": "#B279A2",
}
METRICS = {
    "chrf": ("chrF", "Score (0–100)"),
    "bertscore_f1": ("BERTScore F1", "Score"),
    "comet": ("COMET", "Score"),
}
GROUP_ORDER = [
    ("custom", "mistral"),
    ("custom", "qwen"),
    ("ess", "mistral"),
    ("ess", "qwen"),
]


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")


def save_figure(fig, output_dir: Path, stem: str, formats: list[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def group_label(dataset: str, model: str) -> str:
    return f"{dataset.capitalize()}\n{model.capitalize()}"


def plot_combined_scores(summary: pd.DataFrame, output_dir: Path, formats: list[str]):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    x = np.arange(len(GROUP_ORDER))
    width = 0.34

    for axis, (metric, (title, ylabel)) in zip(axes, METRICS.items()):
        for offset, stage in ((-width / 2, "raw"), (width / 2, "corrected")):
            values = []
            for dataset, model in GROUP_ORDER:
                row = summary[
                    (summary["dataset"] == dataset)
                    & (summary["model"] == model)
                    & (summary["stage"] == stage)
                ]
                if len(row) != 1:
                    raise ValueError(f"Missing summary row for {dataset}/{model}/{stage}")
                values.append(row.iloc[0][f"combined_{metric}_mean"])
            axis.bar(
                x + offset,
                values,
                width,
                label=stage.capitalize(),
                color=COLORS[stage],
            )
        axis.set_title(f"Combined {title}")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [group_label(*group) for group in GROUP_ORDER])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 100 if metric == "chrf" else 1)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Raw and corrected question–scale scores", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, output_dir, "combined_scores", formats)


def plot_combined_deltas(deltas: pd.DataFrame, output_dir: Path, formats: list[str]):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    labels = [group_label(*group) for group in GROUP_ORDER]
    x = np.arange(len(labels))

    for axis, (metric, (title, _)) in zip(axes, METRICS.items()):
        values = []
        colors = []
        for dataset, model in GROUP_ORDER:
            row = deltas[
                (deltas["dataset"] == dataset) & (deltas["model"] == model)
            ]
            if len(row) != 1:
                raise ValueError(f"Missing delta row for {dataset}/{model}")
            value = row.iloc[0][f"combined_{metric}_delta"]
            values.append(value)
            colors.append("#59A14F" if value >= 0 else "#E15759")
        bars = axis.bar(x, values, color=colors, width=0.62)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Combined {title} Δ")
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.margins(y=0.18)
        for bar, value in zip(bars, values):
            precision = 3 if metric == "chrf" else 4
            axis.annotate(
                f"{value:+.{precision}f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle("Correction effect (Step 3 minus Step 1)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, output_dir, "combined_deltas", formats)


def plot_question_scale_deltas(
    deltas: pd.DataFrame, output_dir: Path, formats: list[str]
):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    labels = [group_label(*group) for group in GROUP_ORDER]
    x = np.arange(len(labels))
    width = 0.34

    for axis, (metric, (title, _)) in zip(axes, METRICS.items()):
        for offset, component in ((-width / 2, "question"), (width / 2, "scale")):
            values = []
            for dataset, model in GROUP_ORDER:
                row = deltas[
                    (deltas["dataset"] == dataset) & (deltas["model"] == model)
                ]
                values.append(row.iloc[0][f"{component}_{metric}_delta"])
            axis.bar(
                x + offset,
                values,
                width,
                label=component.capitalize(),
                color=COLORS[component],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{title} Δ by component")
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Question and response-scale correction effects", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, output_dir, "question_scale_deltas", formats)


def infer_model_and_stage(input_file: str) -> tuple[str, str]:
    name = Path(input_file).name.lower()
    model = "mistral" if "mistral" in name else "qwen"
    if "step4" in name or "oracle" in name:
        stage = "Oracle"
    elif "step3" in name:
        stage = "Step 3"
    else:
        stage = "Step 1"
    return model, stage


def plot_trap_avoidance(
    trap_summary_path: Path,
    oracle_summary_path: Path,
    output_dir: Path,
    formats: list[str],
):
    with trap_summary_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    with oracle_summary_path.open("r", encoding="utf-8") as handle:
        records.extend(json.load(handle))

    lookup = {}
    for record in records:
        model, stage = infer_model_and_stage(record["input_file"])
        lookup[(model, stage)] = record["pass_rate_valid_percent"]

    stages = ["Step 1", "Step 3", "Oracle"]
    x = np.arange(len(stages))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for offset, model in ((-width / 2, "mistral"), (width / 2, "qwen")):
        values = [lookup[(model, stage)] for stage in stages]
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=model.capitalize(),
            color=COLORS[model],
        )
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)

    axis.set_xticks(x, stages)
    axis.set_ylabel("Complete traps avoided (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Custom-set trap avoidance", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_dir, "trap_avoidance", formats)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument(
        "--deltas", type=Path, default=Path("results/summary_with_delta.csv")
    )
    parser.add_argument(
        "--trap-summary", type=Path, default=Path("results/trap_checks/summary.json")
    )
    parser.add_argument(
        "--oracle-summary",
        type=Path,
        default=Path("results/oracle_trap_checks/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "pdf"], default=["png", "pdf"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.summary, args.deltas, args.trap_summary, args.oracle_summary):
        require_file(path)
    configure_style()
    summary = pd.read_csv(args.summary)
    deltas = pd.read_csv(args.deltas)
    plot_combined_scores(summary, args.output_dir, args.formats)
    plot_combined_deltas(deltas, args.output_dir, args.formats)
    plot_question_scale_deltas(deltas, args.output_dir, args.formats)
    plot_trap_avoidance(
        args.trap_summary, args.oracle_summary, args.output_dir, args.formats
    )
    print(f"Created report plots in {args.output_dir}")


if __name__ == "__main__":
    main()
