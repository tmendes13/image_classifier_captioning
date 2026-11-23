"""Evaluation metrics.

Classification (Part 2): `accuracy`, `confusion_matrix`, `per_class_accuracy`, on tensors of
predicted and gold class indices.

Captioning (Part 3): `bleu4`, `class_mention_accuracy`, `distinct_ngrams`, on captions given as
lists of word tokens. Tokenize the references with your own tokenizer, and remove special tokens
(<bos>, <eos>, <pad>) from the generated captions before calling these functions.
"""

import math
from collections import Counter

import torch

from load_data import CLASSES


# ----------------------------------------------------------------------------- classification
def accuracy(preds, labels):
    """Fraction of correct predictions. preds, labels: long tensors (N,)."""
    return (preds == labels).float().mean().item()


def confusion_matrix(preds, labels, num_classes=len(CLASSES)):
    """Long tensor (num_classes, num_classes): entry [i, j] counts the examples of gold class i
    predicted as class j."""
    preds, labels = preds.cpu().long(), labels.cpu().long()
    counts = torch.bincount(labels * num_classes + preds, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes)


def per_class_accuracy(preds, labels, num_classes=len(CLASSES)):
    """Float tensor (num_classes,): the accuracy on the examples of each gold class."""
    matrix = confusion_matrix(preds, labels, num_classes)
    return matrix.diag().float() / matrix.sum(dim=1).clamp(min=1)


# ----------------------------------------------------------------------------- captioning
def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def bleu4(hypotheses, references):
    """Corpus-level BLEU-4 (Papineni et al., 2002), between 0 and 100.

    Args:
        hypotheses: list of N generated captions, each a list of tokens
        references: list of N lists of reference captions (5 per image), each a list of tokens

    Clipped n-gram precisions (n = 1..4) are summed over the whole corpus, combined with a
    geometric mean, and multiplied by a brevity penalty that uses, for each caption, the
    reference length closest to it. Same result as nltk's corpus_bleu (no smoothing).
    """
    assert len(hypotheses) == len(references)
    matches, totals = [0] * 4, [0] * 4
    hyp_length, ref_length = 0, 0
    for hyp, refs in zip(hypotheses, references):
        hyp_length += len(hyp)
        ref_length += len(min(refs, key=lambda ref: (abs(len(ref) - len(hyp)), len(ref))))
        for n in range(1, 5):
            hyp_counts = _ngrams(hyp, n)
            max_ref_counts = Counter()
            for ref in refs:
                max_ref_counts |= _ngrams(ref, n)      # max count of each n-gram in any reference
            matches[n - 1] += sum((hyp_counts & max_ref_counts).values())   # clipped counts
            totals[n - 1] += sum(hyp_counts.values())
    if hyp_length == 0 or min(matches) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / 4
    brevity_penalty = min(1.0, math.exp(1 - ref_length / hyp_length))
    return 100 * brevity_penalty * math.exp(log_precision)


# Words that count as mentioning each class; the plural with an "s" is added automatically.
# You may extend this list (say so in your report).
CLASS_SYNONYMS = {
    "airplane": ["airplane", "aeroplane", "plane", "jet", "jetliner", "airliner", "aircraft"],
    "bus": ["bus", "buses", "busses"],
    "cat": ["cat", "kitten", "kitty", "kitties"],
    "dog": ["dog", "puppy", "puppies", "pup"],
    "elephant": ["elephant"],
    "giraffe": ["giraffe"],
    "horse": ["horse", "pony", "ponies", "foal"],
    "motorcycle": ["motorcycle", "motorbike", "bike", "scooter", "moped"],
    "train": ["train", "locomotive"],
    "truck": ["truck", "lorry", "lorries"],
}


def class_mention_accuracy(hypotheses, labels):
    """Fraction of captions that contain a word naming the class of their image.

    Args:
        hypotheses: list of N captions, each a list of lowercase tokens
        labels: long tensor (N,) or list of the images' class indices
    """
    labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
    assert len(hypotheses) == len(labels)
    hits = 0
    for hyp, label in zip(hypotheses, labels):
        names = {word + suffix for word in CLASS_SYNONYMS[CLASSES[label]] for suffix in ("", "s")}
        hits += any(token in names for token in hyp)
    return hits / len(hypotheses)


def distinct_ngrams(hypotheses, n):
    """Number of distinct n-grams over all captions (n = 1: distinct words), a measure of the
    diversity of the generated captions."""
    return len({ngram for hyp in hypotheses for ngram in _ngrams(hyp, n)})
