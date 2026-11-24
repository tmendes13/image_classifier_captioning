"""Q2.3 only: controlled MLP ablations, validation selection and final test analysis.

Reuses models.MLP, common.train_classifier and the supplied training loop unchanged.
Run: python q2_3_study.py --config results/q2_2/best_config.json
"""

import argparse
import copy
import hashlib
import json
import math
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from common import build_mlp, evaluate, mean_std, predict, train_classifier
from data import ImageDataset, channel_stats
from load_data import CLASSES, DEFAULT_ROOT, REVISION, load_coco_dominant10
from metrics import confusion_matrix, per_class_accuracy
from q2_3_report import write_outputs
from train import get_device

HERE = Path(__file__).resolve().parent
PROTOCOL_VERSION = 1


def save_json(value, path):
    """Replace the results file only after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def experiment_configs(selected, epochs):
    """Each variant differs from the reference in exactly one design choice."""
    base = copy.deepcopy(selected)
    base.update(epochs=epochs, patience=None)
    widths = base["hidden_sizes"]
    if len(widths) < 2 or len(set(widths)) != 1 or widths[0] < 2:
        raise ValueError("This study expects a uniform-width MLP with at least two hidden layers")
    if base["dropout"] <= 0:
        raise ValueError("The selected Q2.2 model must have dropout > 0 for this ablation")
    depth = dict(base, hidden_sizes=widths[:-1])
    width = dict(base, hidden_sizes=[widths[0] // 2] * len(widths))
    dropout = dict(base, dropout=0.0)
    return {"reference": base, "less_depth": depth, "less_width": width,
            "no_dropout": dropout}


def assert_controlled(configs):
    base = configs["reference"]
    for name, changed in (("less_depth", "hidden_sizes"), ("less_width", "hidden_sizes"),
                          ("no_dropout", "dropout")):
        assert {k for k in base if base[k] != configs[name][k]} == {changed}
    assert all(c["epochs"] == base["epochs"] and c["patience"] is None
               for c in configs.values())


def independent_loader(dataset, batch_size, seed, shuffle=False):
    """Data order has its own RNG, independent of model size and dropout draws."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def loaders_for_seed(datasets, batch_size, seed):
    train_loader = independent_loader(datasets["train"], batch_size, seed + 1729, True)
    val_loader = independent_loader(datasets["validation"], batch_size, seed + 2718)
    train_eval = independent_loader(datasets["train"], 256, seed + 3141)
    # No test loader is passed into the ablation study.
    return (train_loader, val_loader, None), train_eval


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize(runs):
    """Checkpoint metrics (rather than last-epoch metrics), over the same seeds."""
    result = {"config": runs[0]["config"], "seeds": [r["seed"] for r in runs],
              "parameters": runs[0]["history"]["num_parameters"],
              "best_epochs": [r["history"]["best_epoch"] for r in runs],
              "runs": runs}
    for key in ("train_acc", "val_acc", "train_loss", "val_loss", "gap"):
        values = [r["checkpoint_metrics"][key] for r in runs]
        m, s = mean_std(values)
        result[key] = {"values": values, "mean": m, "std": s}
    return result


def run_one(name, config, seed, datasets, weights_dir, signature, device, verbose):
    weights_path = weights_dir / f"{name}_seed_{seed}.pt"
    loaders, train_eval = loaders_for_seed(datasets, config["batch_size"], seed)
    print(f"[{name}, seed {seed}] training {config['epochs']} full epochs", flush=True)
    model, history = train_classifier(config, seed, loaders, train_eval=train_eval,
                                      device=device, verbose=verbose)
    assert history["epochs_run"] == config["epochs"]
    for key in ("train_loss", "val_loss", "val_score", "train_eval_loss", "train_eval_acc"):
        if not all(math.isfinite(v) for v in history[key]):
            raise RuntimeError(f"Non-finite metric in {name}, seed {seed}: {key}")
    # train_classifier has restored the epoch with the highest validation accuracy.
    train_loss, train_acc = evaluate(model, train_eval)
    val_loss, val_acc = evaluate(model, loaders[1])
    assert abs(val_acc - max(history["val_score"])) < 1e-10
    run = {"signature": signature, "config": config, "seed": seed, "history": history,
           "checkpoint_metrics": {"train_loss": train_loss, "train_acc": train_acc,
                                  "val_loss": val_loss, "val_acc": val_acc,
                                  "gap": train_acc - val_acc}}
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = weights_path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    temporary.replace(weights_path)
    print(f"[{name}, seed {seed}] best epoch {history['best_epoch']}, "
          f"train {100*train_acc:.2f}%, val {100*val_acc:.2f}%", flush=True)
    return run


def final_test(selected_name, summary, data, mean, std, weights_dir, signature, device):
    """Only the validation-selected configuration is evaluated on test."""
    test_signature = fingerprint({"study": signature, "selected": selected_name,
                                  "seeds": summary["seeds"],
                                  "weights": [hashlib.sha256((weights_dir /
                                      f"{selected_name}_seed_{s}.pt").read_bytes()).hexdigest()
                                              for s in summary["seeds"]]})
    test_dataset = ImageDataset(data["test"]["images"], data["test"]["labels"], mean, std)
    loader = independent_loader(test_dataset, 256, 0)
    labels = data["test"]["labels"]
    runs = []
    for seed in summary["seeds"]:
        model = build_mlp(summary["config"]).to(device)
        model.load_state_dict(torch.load(weights_dir / f"{selected_name}_seed_{seed}.pt",
                                         map_location=device, weights_only=True))
        predictions = predict(model, loader)
        matrix = confusion_matrix(predictions, labels)
        class_acc = per_class_accuracy(predictions, labels)
        acc = (predictions == labels).double().mean().item()
        assert matrix.sum().item() == len(labels)
        assert abs(matrix.diag().sum().item() / len(labels) - acc) < 1e-12
        runs.append({"seed": seed, "accuracy": acc, "predictions": predictions.tolist(),
                     "per_class_accuracy": class_acc.tolist(), "confusion_matrix": matrix.tolist()})
        print(f"[test, {selected_name}, seed {seed}] accuracy {100*acc:.2f}%", flush=True)
    matrices = torch.tensor([r["confusion_matrix"] for r in runs], dtype=torch.float64)
    class_accs = torch.tensor([r["per_class_accuracy"] for r in runs], dtype=torch.float64)
    accs = torch.tensor([r["accuracy"] for r in runs], dtype=torch.float64)
    matrix_mean = matrices.mean(0)
    row_percent = 100 * matrix_mean / matrix_mean.sum(1, keepdim=True).clamp(min=1)
    off_diagonal = matrix_mean.clone()
    off_diagonal.fill_diagonal_(0)
    top = off_diagonal.flatten().argsort(descending=True)[:5].tolist()
    confusions = [{"true": CLASSES[k // len(CLASSES)], "predicted": CLASSES[k % len(CLASSES)],
                   "mean_count": off_diagonal.flatten()[k].item()} for k in top]
    result = {"signature": test_signature, "selected": selected_name, "runs": runs,
              "accuracy_mean": accs.mean().item(), "accuracy_std": accs.std().item(),
              "class_accuracy_mean": class_accs.mean(0).tolist(),
              "class_accuracy_std": class_accs.std(0).tolist(),
              "confusion_mean_counts": matrix_mean.tolist(),
              "confusion_row_percent": row_percent.tolist(), "top_confusions": confusions}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "results/q2_2/best_config.json")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--out", type=Path, default=HERE / "results/q2_3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.report_only:
        write_outputs(json.loads((args.out / "results.json").read_text(encoding="utf-8")), args.out)
        return
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("At least three distinct seeds are required")
    selected = json.loads(args.config.read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else selected["epochs"]
    if epochs < 1:
        parser.error("--epochs must be positive")
    configs = experiment_configs(selected, epochs)
    assert_controlled(configs)
    device = get_device()
    # Deterministic kernels where available; this is a CPU-friendly study.
    torch.use_deterministic_algorithms(True)
    data = load_coco_dominant10(str(args.data_root))
    mean, std = channel_stats(data["train"]["images"])
    datasets = {split: ImageDataset(data[split]["images"], data[split]["labels"], mean, std)
                for split in ("train", "validation")}
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ("common.py", "models.py", "train.py", "data.py", "metrics.py",
                                  "load_data.py", "q2_3_study.py")}
    protocol = {"version": PROTOCOL_VERSION, "q2_2_config": selected, "configs": configs,
                "seeds": args.seeds, "dataset_revision": REVISION,
                "dataset_cache_sha256": hashlib.sha256((args.data_root /
                    "coco_dominant10_64.pt").read_bytes()).hexdigest(),
                "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                "source_hashes": source_hashes, "torch": str(torch.__version__),
                "python": platform.python_version(), "device": str(device),
                "threads": args.threads, "shuffle_seed_offset": 1729,
                "selection": "highest mean best-checkpoint validation accuracy; ties: fewer parameters"}
    signature = fingerprint(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    # Weights are needed until validation selection, but are not submission files.
    with TemporaryDirectory(prefix="q2_3_weights_") as temporary:
        weights_dir = Path(temporary)
        for name, config in configs.items():
            runs = [run_one(name, config, seed, datasets, weights_dir, signature,
                            device, args.verbose) for seed in args.seeds]
            summaries[name] = summarize(runs)
        baseline = summaries["reference"]["val_acc"]["values"]
        for summary in summaries.values():
            deltas = [a - b for a, b in zip(summary["val_acc"]["values"], baseline)]
            m, s = mean_std(deltas)
            summary["paired_delta"] = {"values": deltas, "mean": m, "std": s}
        winner = max(summaries, key=lambda name:
                     (summaries[name]["val_acc"]["mean"], -summaries[name]["parameters"]))
        print(f"Selected on validation: {winner}", flush=True)
        test = final_test(winner, summaries[winner], data, mean, std, weights_dir,
                          signature, device)
    results = {"protocol": protocol, "experiments": summaries, "selected": winner,
               "classes": list(CLASSES), "test": test}
    save_json(results, args.out / "results.json")
    write_outputs(results, args.out)
    print(f"Completed Q2.3. Test: {100*test['accuracy_mean']:.2f} +/- "
          f"{100*test['accuracy_std']:.2f}%. Outputs: {args.out}", flush=True)


if __name__ == "__main__":
    main()
