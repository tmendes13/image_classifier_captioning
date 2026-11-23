"""PyTorch datasets and data loaders for COCO-Dominant10-64.

Part 2 (classification), one item per image:

    data = load_coco_dominant10()
    mean, std = channel_stats(data["train"]["images"])        # training set only
    train_loader, val_loader, test_loader = image_loaders(data, batch_size=64, mean=mean, std=std)
    for images, labels in train_loader:                        # (B, 3, 64, 64) float, (B,) long
        ...

Part 3 (captioning), one item per (image, caption) pair. You write `encode`, which maps a caption
string to a list of token ids (with your <bos> and <eos>), and choose the id of your <pad> token:

    train_loader, val_loader, test_loader = caption_loaders(data, encode, pad_id, batch_size=64,
                                                            mean=mean, std=std)
    for images, captions in train_loader:                      # (B, 3, 64, 64) float, (B, T) long
        ...

To generate captions, iterate over `image_loaders` instead (one item per image, in dataset order)
and compare your captions with the references in data[split]["captions"].
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validation", "test")


def channel_stats(images):
    """Per-channel mean and standard deviation, shape (3,), of uint8 images (N, 3, H, W) scaled
    to [0, 1]. Compute them on the training set only."""
    x = images.float() / 255.0
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


class ImageDataset(Dataset):
    """One item per image: (image, label).

    Args:
        images: uint8 tensor (N, 3, H, W), as returned by load_coco_dominant10
        labels: long tensor (N,)
        mean, std: per-channel statistics (see channel_stats) used to standardize the pixels;
            if None, pixels are only scaled to [0, 1]
        flip: if True, flip each image horizontally with probability 1/2 every time it is
            accessed (data augmentation; use it on the training set only)
    """

    def __init__(self, images, labels, mean=None, std=None, flip=False):
        x = images.float() / 255.0
        if mean is not None:
            x = (x - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
        self.images = x
        self.labels = labels
        self.flip = flip

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.flip and torch.rand(()) < 0.5:
            image = image.flip(-1)
        return image, self.labels[i]


class CaptionDataset(Dataset):
    """One item per (image, caption) pair: (image, caption ids). Every image has 5 captions, so it
    appears 5 times per epoch.

    Args:
        image_dataset: an ImageDataset (its normalization and flips are reused)
        captions: list of N lists of caption strings, e.g. data["train"]["captions"]
        encode: your function mapping a caption string to a list of token ids
    """

    def __init__(self, image_dataset, captions, encode):
        self.image_dataset = image_dataset
        self.pairs = [(i, torch.tensor(encode(caption), dtype=torch.long))
                      for i, image_captions in enumerate(captions) for caption in image_captions]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, p):
        i, caption = self.pairs[p]
        image, _ = self.image_dataset[i]
        return image, caption


class PadCollate:
    """collate_fn for CaptionDataset: stacks the images and right-pads the captions with `pad_id`
    to the length of the longest caption in the mini-batch. Returns images (B, 3, H, W) and
    captions (B, T)."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        images, captions = zip(*batch)
        return torch.stack(images), pad_sequence(captions, batch_first=True, padding_value=self.pad_id)


def image_datasets(data, mean=None, std=None, flip=False):
    """{split: ImageDataset}; flips (if any) are applied to the training set only."""
    return {split: ImageDataset(data[split]["images"], data[split]["labels"], mean, std,
                                flip=flip and split == "train")
            for split in SPLITS}


def image_loaders(data, batch_size, mean=None, std=None, flip=False):
    """(train, validation, test) DataLoaders of (images, labels) mini-batches. Only the training
    loader is shuffled."""
    datasets = image_datasets(data, mean, std, flip)
    return tuple(DataLoader(datasets[split], batch_size=batch_size, shuffle=split == "train")
                 for split in SPLITS)


def caption_loaders(data, encode, pad_id, batch_size, mean=None, std=None, flip=False):
    """(train, validation, test) DataLoaders of (images, padded captions) mini-batches, one item
    per (image, caption) pair. Use them to train with teacher forcing and to compute the loss and
    perplexity. Only the training loader is shuffled."""
    datasets = image_datasets(data, mean, std, flip)
    collate = PadCollate(pad_id)
    return tuple(DataLoader(CaptionDataset(datasets[split], data[split]["captions"], encode),
                            batch_size=batch_size, shuffle=split == "train", collate_fn=collate)
                 for split in SPLITS)
