"""Shared classification, evaluation and result-formatting helpers."""

import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ImageDataset, channel_stats, image_loaders
from models import MLP
from train import count_parameters, get_device, set_seed, train


@torch.no_grad()
def predict(model, loader):
    """Return predicted class indices in loader order."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    preds = torch.cat([model(x.to(device)).argmax(-1).cpu() for x, _ in loader])
    model.train(was_training)
    return preds


def mean_std(values):
    """Mean and (sample, n-1) standard deviation of a list of floats, computed with torch."""
    t = torch.tensor(values, dtype=torch.float64)
    std = t.std().item() if len(values) > 1 else 0.0
    return t.mean().item(), std


def fmt_mean_std(values, scale=100.0, digits=1):
    """'mean ± std' string, by default as a percentage (e.g. accuracies)."""
    m, s = mean_std(values)
    return f"{scale * m:.{digits}f} ± {scale * s:.{digits}f}"


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def stack_histories(histories, key):
    """Stack per-seed curves up to the last epoch shared by all runs."""
    length = min(len(h[key]) for h in histories)
    return torch.tensor([h[key][:length] for h in histories], dtype=torch.float64)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt_mean_std_tex(values, scale=100.0, digits=1):
    """Same as fmt_mean_std, with the LaTeX $\\pm$ (for the table_*.tex files)."""
    return fmt_mean_std(values, scale, digits).replace("±", r"$\pm$")


def classification_loaders(data, batch_size, flip=False):
    """Build the three loaders using normalization statistics from the training set only."""
    mean, std = channel_stats(data["train"]["images"])
    return image_loaders(data, batch_size=batch_size, mean=mean, std=std, flip=flip)


def train_eval_loader(data, batch_size=256):
    """Measure training performance without augmentation or shuffling."""
    mean, std = channel_stats(data["train"]["images"])
    dataset = ImageDataset(data["train"]["images"], data["train"]["labels"], mean, std)
    # Even an unshuffled loader draws a seed. Isolate it from the training RNG.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      generator=torch.Generator().manual_seed(0))


@torch.no_grad()
def evaluate(model, loader):
    """Mean cross-entropy and accuracy of `model` over `loader`, in eval mode (dropout off)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros_like(total_loss)
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").double()
        correct += (logits.argmax(-1) == labels).sum()
        count += len(labels)
    model.train(was_training)
    return (total_loss / count).item(), (correct / count).item()


def build_mlp(config):
    """The MLP described by `config` (see models.MLP)."""
    return MLP(hidden_sizes=config["hidden_sizes"], activation=config.get("activation", "relu"),
               dropout=config.get("dropout", 0.0))


def make_optimizer(config, parameters):
    """Build Adam or SGD from the training configuration."""
    lr, weight_decay = config["lr"], config.get("weight_decay", 0.0)
    if config["optimizer"] == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=config.get("momentum", 0.0),
                               weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {config['optimizer']}")


def make_scheduler(config, optimizer):
    """Return the cosine scheduler, or None for a constant learning rate."""
    schedule = config.get("schedule", "constant")
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    raise ValueError(f"unknown schedule {schedule}")


def train_classifier(config, seed, loaders, build_model=build_mlp, train_eval=None, device=None,
                     verbose=False):
    """Return the best-validation model and its per-epoch training history."""
    train_loader, val_loader, _ = loaders
    set_seed(seed)  # Seed before construction so initialization is reproducible.
    model = build_model(config)
    optimizer = make_optimizer(config, model.parameters())
    scheduler = make_scheduler(config, optimizer)
    extra = {"val_loss": [], "train_eval_loss": [], "train_eval_acc": []}

    def loss_fn(model, batch):
        images, labels = batch
        return F.cross_entropy(model(images), labels)

    def val_score_fn(model):
        val_loss, val_acc = evaluate(model, val_loader)
        extra["val_loss"].append(val_loss)
        if train_eval is not None:
            train_loss, train_acc = evaluate(model, train_eval)
            extra["train_eval_loss"].append(train_loss)
            extra["train_eval_acc"].append(train_acc)
        return val_acc

    history = train(model, optimizer, loss_fn, train_loader, val_score_fn,
                    epochs=config["epochs"], scheduler=scheduler, patience=config["patience"],
                    device=device or get_device(), verbose=verbose)
    history.update({k: v for k, v in extra.items() if v})
    history["val_acc_best"] = max(history["val_score"])
    history["epochs_run"] = len(history["val_score"])
    history["num_parameters"] = count_parameters(model)
    return model, history
