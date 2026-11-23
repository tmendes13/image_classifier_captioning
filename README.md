# Deep Learning 2026-27: Course Project skeleton code

This folder contains the code we provide for Parts 2 and 3 of the project: data loading, a
generic training loop, and the evaluation metrics. **Everything else is up to you**: the models
(MLPs, CNNs, RNN decoders), the tokenizer and vocabulary, the loss functions, greedy and beam
search decoding, the evaluation code that calls the metrics, the experiments, and the plots.
You may modify these files, but say what you changed in your report.

## Setup

Python 3.10 or newer, then:

```
pip install -r requirements.txt
python load_data.py
```

The second command downloads the dataset (about 80 MB) into `data/` and prints the size of each
split. On Google Colab, upload this folder (or clone it) and run the same commands in a cell,
with a `!` in front. Part 2 runs fine on a laptop CPU; for Part 3 we recommend a GPU
(e.g., the free Colab GPU: *Runtime > Change runtime type*).

## Files

| File | What it gives you |
|---|---|
| `load_data.py` | `load_coco_dominant10()`: downloads and caches the dataset; returns, for each split, uint8 images `(N, 3, 64, 64)`, labels `(N,)`, 5 caption strings per image, and license metadata. `CLASSES` holds the class names. |
| `data.py` | `channel_stats`, `ImageDataset`, and `image_loaders` for classification; `CaptionDataset`, `PadCollate`, and `caption_loaders` for captioning, which take *your* `encode` function (caption string to token ids) and the id of your padding token. |
| `train.py` | `train(...)`: a training loop with optional learning-rate scheduler, gradient clipping, early stopping, and restoring the best epoch on the validation set; you pass it your `loss_fn` and `val_score_fn`. Also `set_seed`, `count_parameters`, `get_device`. |
| `metrics.py` | `accuracy`, `confusion_matrix`, `per_class_accuracy`; `bleu4`, `class_mention_accuracy` (with the synonym list `CLASS_SYNONYMS`), `distinct_ngrams`. |

## Example

```python
import torch
import torch.nn.functional as F
from load_data import load_coco_dominant10
from data import channel_stats, image_loaders
from metrics import accuracy
from train import set_seed, train

data = load_coco_dominant10()
mean, std = channel_stats(data["train"]["images"])
train_loader, val_loader, test_loader = image_loaders(data, batch_size=64, mean=mean, std=std)

@torch.no_grad()
def predict(model, loader):
    device = next(model.parameters()).device
    return torch.cat([model(x.to(device)).argmax(-1).cpu() for x, _ in loader])

set_seed(0)
model = ...   # your model: images (B, 3, 64, 64) -> class scores (B, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
history = train(model, optimizer,
                loss_fn=lambda model, batch: F.cross_entropy(model(batch[0]), batch[1]),
                train_loader=train_loader,
                val_score_fn=lambda model: accuracy(predict(model, val_loader), data["validation"]["labels"]),
                epochs=20)
```

For captioning, `caption_loaders(data, encode, pad_id, batch_size, mean=mean, std=std)` yields
`(images, captions)` mini-batches, with captions padded to the longest one in the mini-batch.
To generate captions, iterate over `image_loaders` (one item per image, in dataset order) and
compare them with `data[split]["captions"]`, tokenized with your tokenizer.

## Rules (see the project statement)

Allowed: `torch`, `torchvision.transforms`, `matplotlib`, `tqdm`, and this code. Not allowed in
Parts 2 and 3: `numpy`, `scikit-learn`, high-level training frameworks, `torchvision.models`,
`timm`, Hugging Face `transformers`, pretrained weights, and pretrained tokenizers. `pyarrow` and
`Pillow` are used only inside `load_data.py`, to read the data files.
