"""Frozen compact token caches, identical bucket order and decoder shift/padding."""

import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

PAD, BOS, EOS = 0, 1, 2


class TokenDataset(Dataset):
    def __init__(self, directory, split="train", indices=None):
        directory = Path(directory)
        self.source = np.load(directory / f"{split}_source.npy", mmap_mode="r")
        self.target = np.load(directory / f"{split}_target.npy", mmap_mode="r")
        self.source_offsets = np.load(
            directory / f"{split}_source_offsets.npy", mmap_mode="r"
        )
        self.target_offsets = np.load(
            directory / f"{split}_target_offsets.npy", mmap_mode="r"
        )
        self.indices = (
            np.arange(len(self.source_offsets) - 1)
            if indices is None
            else np.asarray(indices)
        )
        self.lengths = (
            np.diff(self.source_offsets)[self.indices]
            + np.diff(self.target_offsets)[self.indices]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        source = self.source[self.source_offsets[row] : self.source_offsets[row + 1]]
        target = self.target[self.target_offsets[row] : self.target_offsets[row + 1]]
        return source.astype(np.int64), target.astype(np.int64), index


class LengthBuckets(Sampler):
    def __init__(self, dataset, batch_size, seed, epoch=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.permutation(len(self.dataset))
        batches = []
        for start in range(0, len(indices), self.batch_size * 50):
            chunk = indices[start : start + self.batch_size * 50]
            chunk = chunk[np.argsort(self.dataset.lengths[chunk], kind="stable")]
            batches.extend(
                chunk[i : i + self.batch_size].tolist()
                for i in range(0, len(chunk), self.batch_size)
            )
        rng.shuffle(batches)
        yield from batches


def collate(rows):
    source_width = math.ceil(max(len(row[0]) for row in rows) / 8) * 8
    target_width = math.ceil(max(len(row[1]) for row in rows) / 8) * 8
    source = torch.full((len(rows), source_width), PAD, dtype=torch.long)
    labels = torch.full((len(rows), target_width), -100, dtype=torch.long)
    for index, (src, target, _) in enumerate(rows):
        source[index, : len(src)] = torch.from_numpy(src)
        labels[index, : len(target)] = torch.from_numpy(target)
    decoder = torch.full_like(labels, PAD)
    decoder[:, 0] = BOS
    decoder[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, PAD)
    return {
        "input_ids": source,
        "attention_mask": source.ne(PAD),
        "labels": labels,
        "decoder_input_ids": decoder,
        "decoder_attention_mask": decoder.ne(PAD),
        "_indices": np.array([row[2] for row in rows], dtype=np.int64),
    }


def train_loader(dataset, batch_size, seed, epoch):
    return DataLoader(
        dataset,
        batch_sampler=LengthBuckets(dataset, batch_size, seed, epoch),
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed + epoch),
    )
