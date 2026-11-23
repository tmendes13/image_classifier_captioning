"""Download COCO-Dominant10-64 and return it as PyTorch tensors.

Usage:
    from load_data import load_coco_dominant10, CLASSES
    data = load_coco_dominant10()          # downloads on first call, then uses a cache
    data["train"]["images"]   # uint8 tensor, shape (6500, 3, 64, 64), values in [0, 255]
    data["train"]["labels"]   # int64 tensor, shape (6500,), values in {0, ..., 9}
    data["train"]["captions"] # list of 6500 lists, each with 5 strings
    data["train"]["metadata"] # list of 6500 dicts (license, flickr_url, coco_url)

The parquet files are read with pyarrow and the PNG images are decoded with Pillow;
these two libraries are only used here, to produce PyTorch tensors.
"""

import io
import json
import os
import urllib.request

import pyarrow.parquet as pq
import torch
from PIL import Image

REVISION = "0daf01af17082a903a25bc9ffb65ebdb79f42b5d"   # dataset commit of 2026-11-04
REPO_URL = f"https://huggingface.co/datasets/M1keR/coco-dominant10-64/resolve/{REVISION}/data"
SPLITS = ("train", "validation", "test")
CLASSES = ("airplane", "bus", "cat", "dog", "elephant",
           "giraffe", "horse", "motorcycle", "train", "truck")
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download(root=DEFAULT_ROOT):
    """Download the three parquet files into `root` (skips files that already exist)."""
    os.makedirs(root, exist_ok=True)
    for split in SPLITS:
        path = os.path.join(root, f"{split}.parquet")
        if not os.path.exists(path):
            url = f"{REPO_URL}/{split}-00000-of-00001.parquet"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)      # an interrupted download never leaves `path`


def _decode_image(png_bytes):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return x.view(h, w, 3).permute(2, 0, 1).contiguous()  # (3, H, W)


def _read_split(path):
    table = pq.read_table(path)
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    assert tuple(names) == CLASSES, f"CLASSES does not match the label names in {path}: {names}"
    rows = table.to_pylist()
    return {
        "images": torch.stack([_decode_image(r["image"]["bytes"]) for r in rows]),
        "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long),
        "captions": [list(r["captions"]) for r in rows],
        "metadata": [dict(r["metadata"]) for r in rows],
    }


def load_coco_dominant10(root=DEFAULT_ROOT):
    """Return {split: {"images", "labels", "captions", "metadata"}} for the three splits."""
    cache = os.path.join(root, "coco_dominant10_64.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=True)
    download(root)
    data = {split: _read_split(os.path.join(root, f"{split}.parquet")) for split in SPLITS}
    torch.save(data, cache + ".part")
    os.replace(cache + ".part", cache)            # an interrupted save never leaves `cache`
    return data


if __name__ == "__main__":
    data = load_coco_dominant10()
    for split in SPLITS:
        d = data[split]
        print(f"{split:>10}: images {tuple(d['images'].shape)} {d['images'].dtype}, "
              f"labels {tuple(d['labels'].shape)}, captions {len(d['captions'])}x{len(d['captions'][0])}")
    print("example caption:", data["train"]["captions"][0][0])
