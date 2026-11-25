"""Q2.4 only: inductive biases of MLPs (weights as images, shifted test images, pixel permutation).

Reuses models.MLP, common.train_classifier and the supplied training loop unchanged, and the
configuration selected in Q2.2 (results/q2_2/best_config.json).
Run: python q2_4_inductive.py          (use --quick for a two-epoch check)

Experiments (all with the same seeds, mean +- std over seeds):
  (0) weights: class templates of the linear baseline and first-layer units of the best MLP.
  (i) shifts: models trained on the ORIGINAL images are evaluated on test images shifted by
      1, 2, ... pixels (horizontal zero-fill, horizontal circular, vertical zero-fill).
  (ii) permutation: ONE fixed random permutation of the 64x64 pixel positions (the same for the
      three channels and for train / validation / test) and the best MLP retrained from scratch.
Everything is written to results/q2_4/. The test set is used only for the final numbers.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import (classification_loaders, evaluate, fmt_mean_std, fmt_mean_std_tex, load_json,
                    mean_std, save_json, train_classifier)
from data import ImageDataset, channel_stats
from load_data import CLASSES, load_coco_dominant10
from train import get_device

# Q2.1 linear baseline (mini-batch SGD). Used only if results/q2_1/results.json or
# --linear-config does not provide the training settings that you used in Q2.1.
DEFAULT_LINEAR = {"hidden_sizes": [], "activation": "relu", "dropout": 0.0, "optimizer": "sgd",
                  "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0, "schedule": "constant",
                  "batch_size": 64, "epochs": 20, "patience": None}
SHIFTS = [0, 1, 2, 3, 4, 6, 8]
H = W = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join("results", "q2_2", "best_config.json"))
    p.add_argument("--linear-config", default=None, help="JSON with the Q2.1 training settings")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--perm-seed", type=int, default=1234)
    p.add_argument("--shifts", type=int, nargs="+", default=SHIFTS)
    p.add_argument("--out", default=os.path.join("results", "q2_4"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true", help="2 epochs, to test the code")
    return p.parse_args()


def load_linear_config(path):
    cfg = dict(DEFAULT_LINEAR)
    if path is not None:
        cfg.update(load_json(path))
        source = path
    else:
        source = "built-in default (SGD, lr 0.01, 20 epochs, batch 64)"
        q21 = os.path.join("results", "q2_1", "results.json")
        if os.path.exists(q21):
            saved = load_json(q21).get("config", {})
            used = {k: saved[k] for k in DEFAULT_LINEAR if k in saved}
            cfg.update(used)
            source = f"{q21} (keys: {sorted(used)}; others default)"
    cfg["hidden_sizes"] = []          # the linear baseline has no hidden layer
    return cfg, source


# ----------------------------------------------------------------------------- shifts
def shift_images(x, s, dim=-1, mode="zero"):
    """Shift a batch (N, C, H, W) by s >= 0 pixels along `dim` (-1: horizontal, -2: vertical),
    towards larger indices. 'zero': pixels that leave are dropped and the new ones are 0 (the mean
    colour after normalization). 'circular': pixels wrap around (the setting of Q1.1)."""
    if s == 0:
        return x
    assert 0 < s < x.shape[dim]
    if mode == "circular":
        return torch.roll(x, shifts=s, dims=dim)
    out = torch.zeros_like(x)
    src, dst = [slice(None)] * x.dim(), [slice(None)] * x.dim()
    src[dim], dst[dim] = slice(0, x.shape[dim] - s), slice(s, None)
    out[tuple(dst)] = x[tuple(src)]
    return out


@torch.no_grad()
def accuracy_on(model, x, y, batch_size=500):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    for i in range(0, len(y), batch_size):
        correct += (model(x[i:i + batch_size].to(device)).argmax(-1).cpu()
                    == y[i:i + batch_size]).sum().item()
    return correct / len(y)


def shift_curve(model, x, y, shifts, dim, mode):
    return [accuracy_on(model, shift_images(x, s, dim, mode), y) for s in shifts]


# ----------------------------------------------------------------------------- permutation
def make_permutation(seed, size=H * W):
    """One fixed permutation of the pixel positions."""
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed))


def permute_data(data, perm):
    """Apply the same position permutation to all channels of the uint8 images of all splits."""
    out = {}
    for split in ("train", "validation", "test"):
        im = data[split]["images"]
        n, c = im.shape[:2]
        out[split] = {"images": im.reshape(n, c, -1)[:, :, perm].reshape(n, c, H, W).contiguous(),
                      "labels": data[split]["labels"]}
    return out


# ----------------------------------------------------------------------------- weights as images
def to_image(w):
    """A weight vector of size 3*64*64 as an (H, W, 3) image in [0, 1] (min-max per image)."""
    w = w.detach().cpu().reshape(3, H, W)
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    return w.permute(1, 2, 0).tolist()


def neighbor_correlation(weights):
    """Mean over units of the Pearson correlation between each weight and its right neighbour
    (within a channel): about 0 for spatially unstructured weights, high for smooth patterns."""
    w = weights.detach().cpu().reshape(-1, 3, H, W)
    a, b = w[..., :-1], w[..., 1:]
    a = a - a.mean(dim=(1, 2, 3), keepdim=True)
    b = b - b.mean(dim=(1, 2, 3), keepdim=True)
    num = (a * b).sum(dim=(1, 2, 3))
    den = (a.pow(2).sum(dim=(1, 2, 3)) * b.pow(2).sum(dim=(1, 2, 3))).sqrt()
    return (num / den).mean().item()


def undo_permutation(weights, perm):
    """Weights learned on permuted images, re-expressed on the original pixel layout. The input
    at permuted position j is the original pixel perm[j], so w_orig[perm[j]] = w_perm[j]."""
    inv = torch.argsort(perm)
    return weights.detach().cpu().reshape(-1, 3, H * W)[:, :, inv].reshape(-1, 3 * H * W)


def weights_figure(path, rows):
    """rows: list of (label, [weight vectors], [titles])."""
    ncols = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(1.15 * ncols + 0.5, 1.3 * len(rows) + 0.2),
                             squeeze=False)
    for r, (label, ws, titles) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(ws):
                ax.imshow(to_image(ws[c]))
                if titles:
                    ax.set_title(titles[c], fontsize=6)
        axes[r, 0].text(-0.1, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)


def shift_figure(path, shifts, curves, labels):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for key, label in labels.items():
        y = 100 * curves[key]
        m, s = y.mean(0), (y.std(0) if y.shape[0] > 1 else torch.zeros(y.shape[1]))
        ax.errorbar(shifts, m.tolist(), yerr=s.tolist(), marker="o", ms=4, capsize=3, label=label)
    ax.axhline(10, color="gray", ls=":", lw=0.8)          # chance level (10 balanced classes)
    ax.set_xlabel("shift of the test images (pixels)")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def write_tex(path, columns, header, rows):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"] + [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule",
                                                                       r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    best = load_json(args.config)
    linear, linear_source = load_linear_config(args.linear_config)
    if args.quick:
        for cfg in (best, linear):
            cfg.update(epochs=2, patience=None)
        args.out = os.path.join("results", "q2_4_quick")
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print("device:", device, "| seeds:", args.seeds, "| shifts:", args.shifts)
    print("MLP config (Q2.2):", best)
    print("linear config:", linear, "| source:", linear_source)

    data = load_coco_dominant10()
    mean, std = channel_stats(data["train"]["images"])
    x_test = ImageDataset(data["test"]["images"], data["test"]["labels"], mean, std).images
    y_test = data["test"]["labels"]

    # ---- train on the ORIGINAL images, evaluate on shifted test images ----
    models, finals = {}, {}
    curves = {k: [] for k in ("linear_h", "mlp_h", "mlp_h_circ", "mlp_v")}
    for name, cfg in (("linear", linear), ("mlp", best)):
        loaders = classification_loaders(data, cfg["batch_size"])
        models[name], finals[name] = [], []
        for seed in args.seeds:
            model, h = train_classifier(cfg, seed, loaders, device=device)
            test_loss, test_acc = evaluate(model, loaders[2])
            assert abs(test_acc - accuracy_on(model, x_test, y_test)) < 1e-9
            print(f"[{name}, seed {seed}] best epoch {h['best_epoch']}/{h['epochs_run']}, "
                  f"val {100 * h['val_acc_best']:.2f}%, test {100 * test_acc:.2f}%", flush=True)
            models[name].append(model)
            finals[name].append({"seed": seed, "val_acc": h["val_acc_best"], "test_acc": test_acc,
                                 "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"],
                                 "num_parameters": h["num_parameters"]})
            if name == "linear":
                curves["linear_h"].append(shift_curve(model, x_test, y_test, args.shifts, -1, "zero"))
            else:
                curves["mlp_h"].append(shift_curve(model, x_test, y_test, args.shifts, -1, "zero"))
                curves["mlp_h_circ"].append(shift_curve(model, x_test, y_test, args.shifts, -1,
                                                        "circular"))
                curves["mlp_v"].append(shift_curve(model, x_test, y_test, args.shifts, -2, "zero"))
    curves = {k: torch.tensor(v, dtype=torch.float64) for k, v in curves.items()}

    # consistency with the Q2.2 final runs (same config, same seeds)
    q22_path = os.path.join(os.path.dirname(args.config), "results.json")
    if not args.quick and os.path.exists(q22_path):
        q22 = load_json(q22_path)["test_acc"]
        diffs = [abs(q22[str(f["seed"])] - f["test_acc"]) for f in finals["mlp"]
                 if str(f["seed"]) in q22]
        print("max |test acc - Q2.2 test acc| over seeds:", max(diffs) if diffs else "n/a")

    # ---- permutation experiment: retrain the best MLP from scratch on permuted images ----
    perm = make_permutation(args.perm_seed)
    pdata = permute_data(data, perm)
    pmean, pstd = channel_stats(pdata["train"]["images"])
    assert torch.allclose(mean, pmean) and torch.allclose(std, pstd)   # permutation keeps stats
    ploaders = classification_loaders(pdata, best["batch_size"])
    pmodels, pfinals = [], []
    for seed in args.seeds:
        model, h = train_classifier(best, seed, ploaders, device=device)
        test_loss, test_acc = evaluate(model, ploaders[2])
        print(f"[mlp, permuted pixels, seed {seed}] best epoch {h['best_epoch']}/{h['epochs_run']}, "
              f"val {100 * h['val_acc_best']:.2f}%, test {100 * test_acc:.2f}%", flush=True)
        pmodels.append(model)
        pfinals.append({"seed": seed, "val_acc": h["val_acc_best"], "test_acc": test_acc,
                        "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"],
                        "num_parameters": h["num_parameters"]})

    # ---- weights as images (seed of the first run) ----
    lin_w = models["linear"][0].first_layer.weight.detach().cpu()          # (10, 12288): one template per class
    mlp_w = models["mlp"][0].first_layer.weight.detach().cpu()             # (hidden, 12288)
    perm_w = pmodels[0].first_layer.weight.detach().cpu()
    n_units = 10
    idx = torch.randperm(mlp_w.shape[0], generator=torch.Generator().manual_seed(0))[:n_units]
    perm_idx = torch.randperm(perm_w.shape[0], generator=torch.Generator().manual_seed(0))[:n_units]
    weights_figure(os.path.join(args.out, "weights"), [
        ("linear: class templates", list(lin_w), list(CLASSES)),
        ("MLP, original images", [mlp_w[i] for i in idx], None),
        ("MLP, permuted pixels (as is)", [perm_w[i] for i in perm_idx], None),
        ("MLP, permuted: permutation undone", list(undo_permutation(perm_w[perm_idx], perm)), None),
    ])
    smooth = {"linear_templates": neighbor_correlation(lin_w),
              "mlp_original": neighbor_correlation(mlp_w),
              "mlp_permuted_as_is": neighbor_correlation(perm_w),
              "mlp_permuted_undone": neighbor_correlation(undo_permutation(perm_w, perm))}
    print("neighbour correlation of the first-layer weights:", smooth)

    # ---- figures, tables, json ----
    shift_figure(os.path.join(args.out, "shift"), args.shifts, curves,
                 {"linear_h": "linear, horizontal (zero-fill)", "mlp_h": "MLP, horizontal (zero-fill)",
                  "mlp_h_circ": "MLP, horizontal (circular)", "mlp_v": "MLP, vertical (zero-fill)"})
    keys = ("linear_h", "mlp_h", "mlp_h_circ", "mlp_v")
    write_tex(os.path.join(args.out, "table_shift.tex"), "r c c c c",
              [r"Shift (px)", "Linear", "MLP", "MLP circ.", "MLP vert."],
              [[str(s)] + [fmt_mean_std_tex(curves[k][:, j].tolist()) for k in keys]
               for j, s in enumerate(args.shifts)])
    val = lambda fs: [f["val_acc"] for f in fs]
    test = lambda fs: [f["test_acc"] for f in fs]
    write_tex(os.path.join(args.out, "table_permutation.tex"), "l c c",
              ["MLP trained on", r"Val. acc. (\%)", r"Test acc. (\%)"],
              [["original pixels", fmt_mean_std_tex(val(finals["mlp"])), fmt_mean_std_tex(test(finals["mlp"]))],
               ["permuted pixels", fmt_mean_std_tex(val(pfinals)), fmt_mean_std_tex(test(pfinals))]])
    paired = [p["test_acc"] - o["test_acc"] for p, o in zip(pfinals, finals["mlp"])]
    save_json({"mlp_config": best, "linear_config": linear, "linear_config_source": linear_source,
               "seeds": args.seeds, "perm_seed": args.perm_seed, "shifts": args.shifts,
               "shift_test_acc": {k: v.tolist() for k, v in curves.items()},
               "original": finals, "permuted": pfinals,
               "permuted_minus_original_test_acc": {"values": paired, "mean_std": mean_std(paired)},
               "first_layer_neighbor_correlation": smooth}, os.path.join(args.out, "results.json"))

    print("\nshift | " + " | ".join(keys))
    for j, s in enumerate(args.shifts):
        print(f"{s:5d} | " + " | ".join(fmt_mean_std(curves[k][:, j].tolist(), digits=1) for k in keys))
    print(f"\nMLP test acc, original pixels: {fmt_mean_std(test(finals['mlp']))}%")
    print(f"MLP test acc, permuted pixels: {fmt_mean_std(test(pfinals))}%")
    print(f"paired difference (permuted - original): {100 * mean_std(paired)[0]:+.2f} points")
    print(f"saved everything to {args.out}/")


if __name__ == "__main__":
    main()
