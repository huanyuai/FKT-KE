from __future__ import annotations

import argparse
import os
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model

import config
from client import Client
from utils.data_utils import create_non_iid_partitions
from torch.utils.tensorboard import SummaryWriter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FKT-KE Federated Orchestration")
    parser.add_argument("--dataset_name", type=str, default="imdb", help="HF dataset name, e.g., 'imdb' or 'ag_news'")
    parser.add_argument("--num_clients", "-K", type=int, default=5, help="Total number of clients")
    parser.add_argument("--num_rounds", type=int, default=20, help="Total federated rounds")
    parser.add_argument("--editor_ckpt_path", type=str, default=None, help="Pretrained editor checkpoint path")
    parser.add_argument("--log_dir", type=str, default="runs", help="Root directory for TensorBoard logs")
    parser.add_argument("--alpha", type=float, default=config.NONIID_ALPHA, help="Dirichlet alpha for non-IID split")
    return parser.parse_args()


def _task_type_from_string(name: Optional[str]) -> TaskType:
    mapping = {
        None: TaskType.SEQ_CLS,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }
    name_up = name.upper() if isinstance(name, str) else None
    return mapping.get(name_up, TaskType.SEQ_CLS)


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=int(config.LORA_CONFIG.get("r", 8)),
        lora_alpha=int(config.LORA_CONFIG.get("lora_alpha", 16)),
        lora_dropout=float(config.LORA_CONFIG.get("lora_dropout", 0.1)),
        bias=str(config.LORA_CONFIG.get("bias", "none")),
        task_type=_task_type_from_string(config.LORA_CONFIG.get("task_type", "SEQ_CLS")),
    )


class TBLossCallback(TrainerCallback):
    def __init__(self, writer: SummaryWriter, tag_prefix: str) -> None:
        self.writer = writer
        self.tag_prefix = tag_prefix
        self._global_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self._global_step = state.global_step
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{self.tag_prefix}/{key}", float(value), self._global_step)


def _compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # predictions may be logits or probabilities
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        y_pred = predictions.argmax(axis=1)
    else:
        # binary single-logit case: threshold at 0
        y_pred = (predictions.ravel() > 0).astype(int)

    y_true = labels.astype(int)
    assert y_true.shape[0] == y_pred.shape[0]

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    # Accuracy
    accuracy = float((y_true == y_pred).mean())

    # Per-class metrics
    recalls = []
    precisions = []
    f1s = []
    eps = 1e-12
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        prec_c = tp / (tp + fp + eps)
        rec_c = tp / (tp + fn + eps)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
        precisions.append(prec_c)
        recalls.append(rec_c)
        f1s.append(f1_c)

    recall_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
    }


def _finetune_client_with_lora(client: Client, output_dir: str, writer: SummaryWriter) -> None:
    # Wrap the client's model with a LoRA adapter
    lora_cfg = _build_lora_config()
    peft_model = get_peft_model(client.model, lora_cfg)

    collator = DataCollatorWithPadding(tokenizer=client.tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.TRAINING_DEFAULTS["per_device_train_batch_size"],
        learning_rate=config.TRAINING_DEFAULTS["learning_rate"],
        num_train_epochs=config.TRAINING_DEFAULTS["num_train_epochs"],
        logging_steps=config.TRAINING_DEFAULTS["logging_steps"],
        max_steps=config.TRAINING_DEFAULTS["max_steps"],
        gradient_accumulation_steps=config.TRAINING_DEFAULTS["gradient_accumulation_steps"],
        save_strategy="no",
        evaluation_strategy="steps",
        eval_steps=config.TRAINING_DEFAULTS["logging_steps"],
        report_to=[],
        remove_unused_columns=False,
    )
    # Split client's private dataset into train/eval
    split = client.private_data.train_test_split(test_size=0.1, seed=config.SEED)
    train_ds = split["train"]
    eval_ds = split["test"]

    def compute_metrics_fn(eval_pred):
        preds, labels = eval_pred
        # Some HF versions provide tuple(preds, None)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return _compute_classification_metrics(preds, labels)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )

    # Attach TB loss logging
    trainer.add_callback(TBLossCallback(writer, tag_prefix=f"client_{client.client_id}"))
    trainer.train()

    # Merge LoRA weights back to the base model for later use in FL
    merged = peft_model.merge_and_unload()
    client.model = merged.to(client.device)


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)

    # Load dataset
    print(f"[Info] Loading dataset: {args.dataset_name}")
    raw = load_dataset(args.dataset_name)
    # Use train split for client private data; keep a small public portion inside partitioner
    base_train: Dataset = raw["train"] if "train" in raw else list(raw.values())[0]

    # Normalize labels field name to 'labels' for Trainer and tokenization convenience
    label_col = "label" if "label" in base_train.column_names else ("labels" if "labels" in base_train.column_names else None)
    if label_col is None:
        raise ValueError("Dataset must contain a label or labels column.")

    # Tokenization function
    def tok_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        text_col = "text"
        if text_col not in batch:
            # Try common names
            for c in ["sentence", "content", "news", "review", "text"]:
                if c in batch:
                    text_col = c
                    break
        toks = tokenizer(batch[text_col], truncation=True, max_length=256)
        # ensure labels field name for Trainer
        if "labels" in batch:
            toks["labels"] = batch["labels"]
        elif "label" in batch:
            toks["labels"] = batch["label"]
        return toks

    tokenized = base_train.map(tok_fn, batched=True, remove_columns=[c for c in base_train.column_names if c not in ["text", "label", "labels"]])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Build non-IID partitions
    print(f"[Info] Creating non-IID partitions: K={args.num_clients}, alpha={args.alpha}")
    private_datasets, public_dataset = create_non_iid_partitions(tokenized, args.num_clients, args.alpha)

    # TensorBoard writers
    os.makedirs(args.log_dir, exist_ok=True)
    writer_global = SummaryWriter(log_dir=os.path.join(args.log_dir, "global"))
    client_writers = [SummaryWriter(log_dir=os.path.join(args.log_dir, f"client_{i}")) for i in range(args.num_clients)]

    # Visualize label distribution per client
    print("[Info] Logging client label distributions to TensorBoard")
    for cid, ds in enumerate(private_datasets):
        labels = ds["labels"].numpy() if hasattr(ds["labels"], "numpy") else np.array(ds["labels"])
        # Use a histogram scalar counts per label
        unique, counts = np.unique(labels, return_counts=True)
        hist_text = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(unique, counts)])
        client_writers[cid].add_text("label_distribution", hist_text, global_step=0)

    # Initialize clients
    clients: List[Client] = []
    for client_id in range(args.num_clients):
        clients.append(Client(client_id=client_id, private_dataset=private_datasets[client_id], device=device, tokenizer=tokenizer))

    # Pre-FL per-client LoRA warm-up
    for client in clients:
        output_dir = os.path.join("./outputs", f"client_{client.client_id}_lora")
        os.makedirs(output_dir, exist_ok=True)
        _finetune_client_with_lora(client, output_dir=output_dir, writer=client_writers[client.client_id])

    print("[Info] LoRA fine-tuning completed for all clients. Ready for FL rounds.")

    # Placeholder for the main federated learning loop
    # for round_idx in range(args.num_rounds):
    #     ...


if __name__ == "__main__":
    main()


