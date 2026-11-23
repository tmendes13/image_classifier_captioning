"""A generic training loop in plain PyTorch, used for both classification and captioning.

You write two functions and pass them to `train`:

    loss_fn(model, batch)  -> scalar loss tensor for one mini-batch, as yielded by your
                              training DataLoader (its tensors are already on the device)
    val_score_fn(model)    -> float, a score on the validation set where higher is better
                              (e.g. accuracy, minus the loss, or BLEU-4)

Example (Part 2):

    set_seed(0)
    model = MyModel(...)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = lambda model, batch: F.cross_entropy(model(batch[0]), batch[1])
    history = train(model, optimizer, loss_fn, train_loader,
                    val_score_fn=lambda model: my_accuracy(model, val_loader), epochs=20)
"""

import copy
import math
import random
import time

import torch
from torch import nn


def set_seed(seed):
    """Seed Python's and PyTorch's random number generators (on CPU and GPU). This also fixes
    the order in which the training DataLoader shuffles the data."""
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch, device):
    """Move the tensors of a mini-batch (a tensor, or a tuple or list of them) to `device`."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(b, device) for b in batch)
    return batch


def train(model, optimizer, loss_fn, train_loader, val_score_fn, epochs, scheduler=None,
          patience=None, grad_clip=None, device=None, verbose=True):
    """Train `model` and restore the parameters of the epoch with the best validation score.

    Args:
        model: an nn.Module; it is moved to `device`
        optimizer: a torch.optim optimizer over the parameters you want to update
        loss_fn: loss_fn(model, batch) -> scalar loss tensor (see the module docstring)
        train_loader: DataLoader of training mini-batches
        val_score_fn: val_score_fn(model) -> float, higher is better; called after every epoch,
            with the model in eval mode and gradients disabled
        epochs: maximum number of epochs
        scheduler: optional learning-rate scheduler, stepped once per epoch
        patience: stop if the validation score has not improved for this many epochs
            (None: always train for `epochs` epochs)
        grad_clip: if not None, clip the gradient norm to this value before each update
        device: defaults to the GPU if there is one, else the CPU
        verbose: print one line per epoch

    Returns:
        history: dict with the lists "train_loss" (mean loss over each epoch's mini-batches),
        "val_score", and "seconds" (per epoch), and "best_epoch" (1-based).

    The model is in train mode while `loss_fn` runs. If part of it is frozen, remember that
    dropout and batch normalization behave differently in train mode: call .eval() on that part
    inside `loss_fn` if needed.
    """
    device = device or get_device()
    model.to(device)
    history = {"train_loss": [], "val_score": [], "seconds": [], "best_epoch": 0}
    best_score, best_state, epochs_without_improvement = -math.inf, None, 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss, num_batches = 0.0, 0
        for batch in train_loader:
            loss = loss_fn(model, to_device(batch, device))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            score = val_score_fn(model)
        history["train_loss"].append(total_loss / num_batches)
        history["val_score"].append(score)
        history["seconds"].append(time.time() - start)

        improved = score > best_score
        if improved:   # keep a copy of the best parameters so far
            best_score, best_state = score, copy.deepcopy(model.state_dict())
            history["best_epoch"], epochs_without_improvement = epoch, 0
        else:
            epochs_without_improvement += 1
        if verbose:
            print(f"epoch {epoch:3d} | train loss {history['train_loss'][-1]:.4f} | "
                  f"val score {score:.4f} | {history['seconds'][-1]:5.1f}s{' *' if improved else ''}")
        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f"no improvement for {patience} epochs: stopping early")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"restored the best epoch ({history['best_epoch']}, val score {best_score:.4f})")
    return history
