"""Q2.3 tables and figures required for the empirical study."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

NAMES = {"reference": "Reference (Q2.2)", "less_depth": "Reduced depth",
         "less_width": "Reduced width", "no_dropout": "Dropout: 0"}


def latex_percent(metric):
    return f"{100*metric['mean']:.2f} $\\pm$ {100*metric['std']:.2f}"


def tex_table(path, columns, header, rows):
    lines = [r"\begin{tabular}{" + columns + "}", r"\hline",
             " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def curves(results, out):
    fig, axes = plt.subplots(len(results["experiments"]), 2, figsize=(12, 13),
                             constrained_layout=True, squeeze=False)
    for row, (name, experiment) in enumerate(results["experiments"].items()):
        histories = [r["history"] for r in experiment["runs"]]
        series = [(0, "train_eval_loss", "train (eval)", "C0", 1),
                  (0, "val_loss", "validation", "C1", 1),
                  (1, "train_eval_acc", "train (eval)", "C0", 100),
                  (1, "val_score", "validation", "C1", 100)]
        for col, key, label, color, scale in series:
            values = scale * torch.tensor([h[key] for h in histories], dtype=torch.float64)
            m, s = values.mean(0), values.std(0)
            epochs = list(range(1, len(m) + 1))
            ax = axes[row, col]
            ax.plot(epochs, m.tolist(), label=label, color=color)
            ax.fill_between(epochs, (m-s).tolist(), (m+s).tolist(), alpha=0.18, color=color)
        for col, ylabel in ((0, "Cross entropy"), (1, "Accuracy (%)")):
            ax = axes[row, col]
            ax.set(title=NAMES[name], xlabel="Epoch", ylabel=ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)


def confusion_plot(results, out):
    names, test = results["classes"], results["test"]
    matrix = test["confusion_row_percent"]
    fig, ax = plt.subplots(figsize=(10.2, 9), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 55 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    n = len(test["runs"])
    ax.set(xlabel="Predicted class", ylabel="True class",
           title=f"Test confusion: mean row percentage over {n} seeds (no ensemble)")
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def write_outputs(results, output):
    """Keep only the two report tables and the two analysis figures."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, e in results["experiments"].items():
        rows.append([NAMES[name], f"{e['parameters']:,}", latex_percent(e["train_acc"]),
                     latex_percent(e["val_acc"]), latex_percent(e["paired_delta"])])
    tex_table(out / "table_ablation.tex", "l r c c c",
              ["Configuration", "Parameters", r"Train (\%)", r"Validation (\%)",
               r"$\Delta$ val. (pp)"], rows)
    test = results["test"]
    tex_table(out / "table_per_class.tex", "l c", ["Class", r"Test accuracy (\%)"],
              [[name, f"{100*m:.2f} $\\pm$ {100*s:.2f}"] for name, m, s in
               zip(results["classes"], test["class_accuracy_mean"], test["class_accuracy_std"])])
    curves(results, out)
    confusion_plot(results, out)
