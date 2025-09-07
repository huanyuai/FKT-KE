from __future__ import annotations

import glob
import os
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset, Dataset
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
        private_data_path: str,
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

        # Load private dataset and tokenize
        self.private_data: Dataset = self._load_and_tokenize_private_dataset(
            private_data_path=private_data_path,
            tokenizer=tokenizer,
        )

        # Placeholder for the knowledge editor (to be provided later)
        self.editor = None

    # ------------------------- Internal helpers ------------------------- #
    def _load_and_tokenize_private_dataset(
        self, private_data_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 256
    ) -> Dataset:
        """
        Load client private data from a file or directory of JSON/JSONL files.
        Expected schema: each record has fields {"text": str, "label": int}

        If nothing is found, fall back to a tiny synthetic dataset so that
        the code path remains runnable for smoke tests.
        """

        dataset: Optional[Dataset] = None

        def _tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, Any]:
            tokenized = tokenizer(
                examples.get("text", [""] * len(examples.get("label", []))),
                truncation=True,
                max_length=max_length,
            )
            # Ensure labels are present under the name expected by Trainer
            if "label" in examples:
                tokenized["labels"] = examples["label"]
            elif "labels" in examples:
                tokenized["labels"] = examples["labels"]
            else:
                tokenized["labels"] = [0] * len(tokenized["input_ids"])  # default dummy labels
            return tokenized

        def _load_from_path(path: str) -> Optional[Dataset]:
            if os.path.isfile(path):
                data_files = {"train": path}
                ds_dict = load_dataset("json", data_files=data_files)
                return ds_dict["train"]

            if os.path.isdir(path):
                files = sorted(
                    glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.jsonl"))
                )
                if files:
                    ds_dict = load_dataset("json", data_files={"train": files})
                    return ds_dict["train"]
            return None

        dataset = _load_from_path(private_data_path)

        if dataset is None or len(dataset) == 0:
            # Fallback synthetic dataset to keep the flow runnable
            dataset = Dataset.from_dict(
                {
                    "text": [
                        "The movie was fantastic and I loved it!",
                        "The product quality was poor and disappointing.",
                    ],
                    "label": [1, 0],
                }
            )

        # Normalize column names if needed
        if "label" not in dataset.column_names and "labels" in dataset.column_names:
            dataset = dataset.rename_column("labels", "label")
        if "text" not in dataset.column_names:
            # Try to pick the first string column as text
            for col in dataset.column_names:
                if isinstance(dataset[0][col], str):
                    dataset = dataset.rename_column(col, "text")
                    break

        tokenized_dataset = dataset.map(_tokenize_batch, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized_dataset


