from __future__ import annotations

import glob
import os
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

import config


class Client:
    """
    Federated client that owns a private non-IID dataset and a local model.

    The client performs a short LoRA-based warm-up before entering the
    federated learning rounds. A shared tokenizer is provided by the server.
    """

    def __init__(
        self,
        client_id: int,
        private_dataset: Dataset,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.client_id: int = client_id
        self.device: torch.device = device
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        # Load base model per config
        # Default to binary classification; callers may later resize as needed
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=2,
        )
        self.model.to(self.device)

        # Save pre-built private dataset (already tokenized by caller)
        self.private_data: Dataset = private_dataset

        # Placeholder for the knowledge editor (to be provided later)
        self.editor = None

    # Placeholder for potential future helpers


