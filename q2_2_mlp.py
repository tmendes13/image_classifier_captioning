"""Tune the Q2.2 MLP on validation accuracy and compare it with the linear baseline.

Run python q2_2_mlp.py to write results, tables and figures to results/q2_2/.
Use --quick for a two-epoch check in results/q2_2_quick/.
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import (classification_loaders, evaluate, fmt_mean_std, fmt_mean_std_tex, load_json,
                    mean_std, save_json, stack_histories, train_classifier, train_eval_loader)
from load_data import load_coco_dominant10
from train import get_device

BASE = {"hidden_sizes": None, "activation": "relu", "dropout": 0.2, "optimizer": "adam",
        "lr": 3e-4, "weight_decay": 0.0, "schedule": "constant", "batch_size": 64,
        "epochs": 30, "patience": 6}
DEPTHS, WIDTHS = [1, 2, 3], [256, 512, 1024]
LRS, DROPOUTS = [1e-4, 3e-4, 1e-3], [0.0, 0.2, 0.5]
TOP_K = 3                              # configurations confirmed with all seeds in stage 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default=os.path.join("results", "q2_2"))
    p.add_argument("--no-cache", action="store_true", help="retrain the search runs already saved")
    p.add_argument("--quick", action="store_true", help="tiny grid and 2 epochs, to test the code")
    return p.parse_args()


def config_name(config):
    """Format the configuration for cache filenames."""
    h = "x".join(map(str, config["hidden_sizes"]))
    return (f"h{h}_{config['activation']}_do{config['dropout']:g}_{config['optimizer']}"
            f"_lr{config['lr']:g}_wd{config['weight_decay']:g}_{config['schedule']}"
            f"_bs{config['batch_size']}_e{config['epochs']}_p{config['patience']}")


def describe(config):
    """Human-readable description, e.g. '2x512, dropout 0.2, lr 3e-4'."""
    hs = config["hidden_sizes"]
    arch = f"{len(hs)}x{hs[0]}" if hs else "linear"
    text = f"{arch}, dropout {config['dropout']:g}, lr {config['lr']:g}"
    return text + (", cosine" if config["schedule"] == "cosine" else "")


class SearchRunner:
    """Reuse or train one validation run per configuration and seed."""

    def __init__(self, loaders, out, use_cache, device):
        self.loaders, self.use_cache, self.device = loaders, use_cache, device
        self.dir = os.path.join(out, "runs")
        self.done = {}                 # runs of this execution (a run can appear in two stages)

    def path(self, config, seed):
        return os.path.join(self.dir, f"{config_name(config)}_s{seed}.json")

    def __call__(self, config, seed, stage):
        path = self.path(config, seed)
        run = self.done.get(path)
        status = "reused"
        if run is None and self.use_cache and os.path.exists(path):
            run = load_json(path)
            status = "cached"
        if run is not None and (run.get("config") != config or run.get("seed") != seed):
            print(f"  WARNING: configuration or seed mismatch in {path}; retraining this run")
            run = None
        if run is None:
            start = time.time()
            _, h = train_classifier(config, seed, self.loaders, device=self.device)
            run = {"config": config, "seed": seed,
                   "history": {k: h[k] for k in ("train_loss", "val_score", "val_loss",
                                                 "seconds", "best_epoch")},
                   "val_acc_best": h["val_acc_best"], "best_epoch": h["best_epoch"],
                   "epochs_run": h["epochs_run"], "num_parameters": h["num_parameters"],
                   "seconds": time.time() - start}
            save_json(run, path)
            status = f"{run['seconds']:.0f}s"
        self.done[path] = run
        print(f"  [{stage}] {describe(config)}, seed {seed}: "
              f"val {100 * run['val_acc_best']:.2f}%, "
              f"epoch {run['best_epoch']}/{run['epochs_run']} ({status})")
        return run


def seed_summary(runs):
    """Mean/std of the best validation accuracy over the seeds of one configuration."""
    vals = [r["val_acc_best"] for r in runs]
    m, s = mean_std(vals)
    return {"config": runs[0]["config"], "name": config_name(runs[0]["config"]),
            "seeds": [r["seed"] for r in runs], "val_accs": vals, "val_mean": m, "val_std": s,
            "best_epochs": [r["best_epoch"] for r in runs],
            "num_parameters": runs[0]["num_parameters"]}


def pick_best(summaries):
    """Highest mean validation accuracy; ties -> fewer parameters (the simpler model)."""
    return max(summaries, key=lambda s: (s["val_mean"], -s["num_parameters"]))


def grid_table(path, row_name, rows, col_name, cols, value, row_fmt, col_fmt):
    """LaTeX table of a 2-D grid of validation accuracies (seed 0)."""
    header = f"{row_name} $\\backslash$ {col_name}"
    lines = [r"\begin{tabular}{l " + "c" * len(cols) + "}", r"\toprule",
             header + " & " + " & ".join(col_fmt(c) for c in cols) + r" \\",
             r"\midrule"]
    for r in rows:
        cells = [f"{100 * value(r, c):.1f}" for c in cols]
        lines.append(" & ".join([row_fmt(r), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def architecture_tex(config):
    widths = config["hidden_sizes"]    # each architecture in the search has uniform width
    return rf"${len(widths)}\times{widths[0]}$"


def heatmap(ax, values, row_labels, col_labels, title, xlabel, ylabel):
    """Plot validation accuracy as a percentage in each grid cell."""
    pct = [[100 * v for v in row] for row in values]
    im = ax.imshow(pct, cmap="viridis")
    middle = (max(map(max, pct)) + min(map(min, pct))) / 2
    for i, row in enumerate(pct):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="black" if v > middle else "white")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="val. acc. (%)")


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def plot_curves(histories, path, title):
    """Plot mean ± sample SD over the epochs shared by all final runs."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    series = [
        (axes[0], "train_loss", "train (mini-batch average, dropout on)", "C0", "--", 1),
        (axes[0], "train_eval_loss", "train (eval mode)", "C0", "-", 1),
        (axes[0], "val_loss", "validation", "C1", "-", 1),
        (axes[1], "train_eval_acc", "train (eval mode)", "C0", "-", 100),
        (axes[1], "val_score", "validation", "C1", "-", 100),
    ]
    for ax, key, label, color, style, scale in series:
        y = scale * stack_histories(histories, key)
        epochs = list(range(1, y.shape[1] + 1))
        m = y.mean(0)
        s = y.std(0) if y.shape[0] > 1 else torch.zeros_like(m)
        ax.plot(epochs, m.tolist(), color=color, ls=style, label=label)
        ax.fill_between(epochs, (m - s).tolist(), (m + s).tolist(), color=color, alpha=0.2)
    for h in histories:                       # best-validation epoch of each seed
        for ax in axes:
            ax.axvline(h["best_epoch"], color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_ylabel("accuracy (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    save_figure(fig, path)


def main():
    args = parse_args()
    global DEPTHS, WIDTHS, LRS, DROPOUTS, TOP_K
    if args.quick:
        DEPTHS, WIDTHS, LRS, DROPOUTS, TOP_K = [1, 2], [64], [1e-3], [0.0, 0.2], 2
        BASE.update(epochs=2, patience=1)
        args.out = os.path.join("results", "q2_2_quick")
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print("device:", device, "| seeds:", args.seeds)

    data = load_coco_dominant10()
    loaders = classification_loaders(data, BASE["batch_size"])
    run = SearchRunner(loaders, args.out, use_cache=not args.no_cache, device=device)
    seed0 = args.seeds[0]

    # Stage 1: architecture (depth x width), seed 0
    print(f"\n== Stage 1: depth and width (seed {seed0}) ==")
    stage1 = {}
    for depth in DEPTHS:
        for width in WIDTHS:
            config = dict(BASE, hidden_sizes=[width] * depth)
            stage1[(depth, width)] = run(config, seed0, "stage1")
    best_arch = max(stage1, key=lambda k: (stage1[k]["val_acc_best"], -stage1[k]["num_parameters"]))
    arch = [best_arch[1]] * best_arch[0]
    print(f"best architecture: {best_arch[0]} hidden layer(s) of {best_arch[1]}")

    # Stage 2: lr x dropout on the best architecture, seed 0
    print(f"\n== Stage 2: lr and dropout ({arch}, seed {seed0}) ==")
    stage2 = {}
    for lr in LRS:
        for dropout in DROPOUTS:
            config = dict(BASE, hidden_sizes=arch, lr=lr, dropout=dropout)
            stage2[(lr, dropout)] = run(config, seed0, "stage2")

    # Stage 3: confirm the top-k configurations of stages 1-2 with every seed
    candidates = {}                            # unique configurations, by name
    for r in list(stage1.values()) + list(stage2.values()):
        candidates[config_name(r["config"])] = r
    ranked = sorted(candidates.values(), key=lambda r: (-r["val_acc_best"], r["num_parameters"]))
    top = ranked[:TOP_K]
    print(f"\n== Stage 3: top {len(top)} configurations of stages 1-2, seeds {args.seeds} ==")
    stage3 = [seed_summary([run(r["config"], s, "stage3") for s in args.seeds]) for r in top]
    winner3 = pick_best(stage3)
    for s in stage3:
        print(f"  {describe(s['config']):<38} val acc {fmt_mean_std(s['val_accs'], digits=2)}%"
              f"{'   <- best' if s is winner3 else ''}")

    # Stage 4: learning-rate schedule (constant vs cosine), every seed
    print(f"\n== Stage 4: schedule ({describe(winner3['config'])}) ==")
    cosine = dict(winner3["config"], schedule="cosine")
    stage4 = seed_summary([run(cosine, s, "stage4") for s in args.seeds])
    selected = pick_best([winner3, stage4])
    for s in (winner3, stage4):
        print(f"  {describe(s['config']):<38} val acc {fmt_mean_std(s['val_accs'], digits=2)}%"
              f"{'   <- selected' if s is selected else ''}")
    best_config = selected["config"]
    save_json(best_config, os.path.join(args.out, "best_config.json"))
    print("selected configuration:", best_config)

    # Final runs: retrain the selected configuration, learning curves, test (once)
    print(f"\n== Final runs: {describe(best_config)}, seeds {args.seeds} (with test evaluation) ==")
    train_eval = train_eval_loader(data)
    finals = []
    for seed in args.seeds:
        model, h = train_classifier(best_config, seed, loaders, train_eval=train_eval,
                                    device=device)
        # Use the test set only after selecting and saving the configuration.
        h["test_loss"], h["test_acc"] = evaluate(model, loaders[2])
        search_history = run.done[run.path(best_config, seed)]["history"]
        checked = ("train_loss", "val_score", "val_loss", "best_epoch")
        differences = [key for key in checked if h[key] != search_history[key]]
        h["search_history_matches"] = not differences
        finals.append(h)
        print(f"  seed {seed}: best epoch {h['best_epoch']}/{h['epochs_run']}, "
              f"val acc {100 * h['val_acc_best']:.2f}%, test acc {100 * h['test_acc']:.2f}%"
              + (f"  WARNING: search history differs in {', '.join(differences)}"
                 if differences else "  (search curves and best epoch match exactly)"))

    val = [h["val_acc_best"] for h in finals]
    test = [h["test_acc"] for h in finals]
    num_params = finals[0]["num_parameters"]

    linear_path = os.path.join("results", "q2_1", "results.json")
    linear = load_json(linear_path) if os.path.exists(linear_path) else None
    print(f"\nMLP: val {fmt_mean_std(val)}%, test {fmt_mean_std(test)}%")
    comparison = None
    seed_keys = [str(seed) for seed in args.seeds]
    if linear is None:
        print("Linear comparison skipped: run q2_1_linear.py first.")
    elif any(key not in linear["val_acc"] or key not in linear["test_acc"] for key in seed_keys):
        print("Linear comparison skipped: Q2.1 results are missing requested seeds.")
    else:
        lin_val = [linear["val_acc"][key] for key in seed_keys]
        lin_test = [linear["test_acc"][key] for key in seed_keys]
        gains = {key: 100 * (score - baseline)
                 for key, score, baseline in zip(seed_keys, test, lin_test)}
        comparison = {"seeds": args.seeds, "linear_val_acc": lin_val, "linear_test_acc": lin_test,
                      "linear_num_parameters": linear["config"]["num_parameters"],
                      "test_gain_points_by_seed": gains,
                      "test_gain_points": 100 * (mean_std(test)[0] - mean_std(lin_test)[0])}
        print(f"Linear: val {fmt_mean_std(lin_val)}%, test {fmt_mean_std(lin_test)}%")
        for seed, gain in gains.items():
            print(f"  seed {seed}: test gain {gain:+.1f} points")
        print(f"Mean paired test gain: {comparison['test_gain_points']:+.1f} points")

    results = {
        "fixed_choices": {k: BASE[k] for k in ("activation", "optimizer", "weight_decay",
                                               "batch_size", "epochs", "patience")},
        "search_space": {"depths": DEPTHS, "widths": WIDTHS, "lrs": LRS, "dropouts": DROPOUTS,
                         "stage1_lr": BASE["lr"], "stage1_dropout": BASE["dropout"],
                         "top_k": TOP_K},
        "selection_criterion": ("best validation accuracy (restored epoch); "
                                "stages 3-4: mean over seeds"),
        "stage1": [{"depth": d, "width": w, "val_acc": r["val_acc_best"],
                    "best_epoch": r["best_epoch"], "num_parameters": r["num_parameters"]}
                   for (d, w), r in stage1.items()],
        "stage1_best_architecture": arch,
        "stage2": [{"lr": lr, "dropout": p, "val_acc": r["val_acc_best"],
                    "best_epoch": r["best_epoch"]} for (lr, p), r in stage2.items()],
        "stage3": stage3,
        "stage4_cosine": stage4,
        "selected_config": best_config,
        "num_parameters": num_params,
        "seeds": args.seeds,
        "val_acc": dict(zip(map(str, args.seeds), val)),
        "test_acc": dict(zip(map(str, args.seeds), test)),
        "val_acc_mean_std": mean_std(val),
        "test_acc_mean_std": mean_std(test),
        "best_epochs": [h["best_epoch"] for h in finals],
        "epochs_run": [h["epochs_run"] for h in finals],
        "final_search_histories_match": all(h["search_history_matches"] for h in finals),
        "comparison_with_linear": comparison,
        "search_runs_distinct": len(run.done),
        "search_seconds_total": sum(r["seconds"] for r in run.done.values()),
        "final_runs_seconds": [sum(h["seconds"]) for h in finals],
        "histories": {str(s): {k: h[k] for k in ("train_loss", "train_eval_loss", "train_eval_acc",
                                                 "val_loss", "val_score", "seconds", "best_epoch",
                                                 "test_acc", "test_loss", "search_history_matches")}
                      for s, h in zip(args.seeds, finals)},
    }
    save_json(results, os.path.join(args.out, "results.json"))

    grid_table(os.path.join(args.out, "table_search_arch.tex"), "Depth", DEPTHS, "width", WIDTHS,
               lambda d, w: stage1[(d, w)]["val_acc_best"], str, str)
    grid_table(os.path.join(args.out, "table_search_lr_dropout.tex"),
               "lr", LRS, "dropout", DROPOUTS,
               lambda lr, p: stage2[(lr, p)]["val_acc_best"],
               lambda lr: f"{lr:g}", lambda p: f"{p:g}")
    lines = [r"\begin{tabular}{@{}l r r l c l@{}}", r"\toprule",
             r"Arch. & Dropout & LR & Schedule & Val. acc. (\%) & Best epochs \\", r"\midrule"]
    for s in stage3 + [stage4]:
        config = s["config"]
        mark = r" $\star$" if s is selected else ""
        epochs = ", ".join(map(str, s["best_epochs"]))
        lines.append(f"{architecture_tex(config)}{mark} & {config['dropout']:g} & "
                     f"{config['lr']:g} & {config['schedule']} & "
                     f"{fmt_mean_std_tex(s['val_accs'])} & {epochs} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(args.out, "table_confirm.tex"), "w") as f:
        f.write("\n".join(lines))
    lines = [r"\begin{tabular}{l r c c}", r"\toprule",
             r"Model & Parameters & Val. acc. (\%) & Test acc. (\%) \\", r"\midrule"]
    if comparison is not None:
        lines.append(f"Linear (Q2.1) & {comparison['linear_num_parameters']:,} & "
                     f"{fmt_mean_std_tex(comparison['linear_val_acc'])} & "
                     f"{fmt_mean_std_tex(comparison['linear_test_acc'])} " + r"\\")
    lines.append(f"MLP ({architecture_tex(best_config)}) & {num_params:,} & "
                 f"{fmt_mean_std_tex(val)} & {fmt_mean_std_tex(test)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(args.out, "table_results.tex"), "w") as f:
        f.write("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    heatmap(axes[0], [[stage1[(d, w)]["val_acc_best"] for w in WIDTHS] for d in DEPTHS],
            DEPTHS, WIDTHS,
            f"Stage 1 (lr {BASE['lr']:g}, dropout {BASE['dropout']:g}, seed {seed0})",
            "width", "depth")
    heatmap(axes[1], [[stage2[(lr, p)]["val_acc_best"] for p in DROPOUTS] for lr in LRS],
            [f"{lr:g}" for lr in LRS], [f"{p:g}" for p in DROPOUTS],
            f"Stage 2 ({len(arch)}x{arch[0]}, seed {seed0})", "dropout", "learning rate")
    save_figure(fig, os.path.join(args.out, "search"))
    plot_curves(finals, os.path.join(args.out, "curves"),
                f"MLP {describe(best_config)}: mean ± std over seeds {args.seeds}")
    print(f"\nsaved everything to {args.out}/")


if __name__ == "__main__":
    main()
