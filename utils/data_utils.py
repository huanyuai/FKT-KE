from __future__ import annotations

import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict

import config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_non_iid_partitions(dataset: Dataset, num_clients: int, alpha: float) -> Tuple[List[Dataset], Dataset]:
    """
    Create non-IID partitions from a labeled HF Dataset using a Dirichlet distribution over labels.

    Args:
        dataset: HF Dataset with columns including 'labels' or 'label'.
        num_clients: Number of clients to split into.
        alpha: Dirichlet concentration parameter. Lower -> more non-IID.

    Returns:
        private_datasets: List of length num_clients with client private datasets.
        public_dataset: A (small) shared dataset sampled uniformly from the full dataset.
    """

    _set_seed(config.SEED)

    # Normalize label column name
    label_col = "label" if "label" in dataset.column_names else "labels"
    if label_col not in dataset.column_names:
        raise ValueError("Dataset must contain a 'label' or 'labels' column for classification tasks.")

    if label_col == "labels":
        # Keep a copy; we won't rename to avoid breaking mapped/tokenized datasets
        labels = dataset[label_col]
    else:
        labels = dataset[label_col]

    labels = np.array(labels)
    unique_labels = np.unique(labels)

    # Build index lists per class label
    label_to_indices: Dict[int, List[int]] = {int(l): [] for l in unique_labels}
    for idx, lab in enumerate(labels):
        label_to_indices[int(lab)].append(idx)

    # For public dataset, sample a small uniform slice from the whole set
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)
    public_count = max(1, int(0.02 * len(dataset)))  # 2% for public, tweak as needed
    public_indices = total_indices[:public_count].tolist()
    public_dataset = dataset.select(public_indices)

    # Remove public indices from label buckets to avoid overlap
    public_set = set(public_indices)
    for lab in unique_labels:
        label_to_indices[int(lab)] = [i for i in label_to_indices[int(lab)] if i not in public_set]

    # Dirichlet-based allocation per class
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for lab in unique_labels:
        idxs = label_to_indices[int(lab)]
        if not idxs:
            continue
        np.random.shuffle(idxs)
        # Sample proportions for this label across clients
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        # Translate proportions into split sizes
        splits = (np.floor(proportions * len(idxs))).astype(int)
        # Adjust for rounding to ensure all indices are assigned
        while splits.sum() < len(idxs):
            splits[np.argmin(splits)] += 1
        while splits.sum() > len(idxs):
            splits[np.argmax(splits)] -= 1

        start = 0
        for client_id in range(num_clients):
            end = start + splits[client_id]
            if end > start:
                client_indices[client_id].extend(idxs[start:end])
            start = end

    # Build HF subsets for each client
    private_datasets: List[Dataset] = []
    for cid in range(num_clients):
        inds = client_indices[cid]
        if len(inds) == 0:
            # Guarantee non-empty by sampling at least one leftover index
            inds = [int(np.random.randint(0, len(dataset)))]
        private_datasets.append(dataset.select(inds))

    return private_datasets, public_dataset


