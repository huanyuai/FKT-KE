from __future__ import annotations

import argparse
import os
import json
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime

import config
from client import Client
from utils.data_utils import create_non_iid_partitions
from torch.utils.tensorboard import SummaryWriter
from utils.eval_utils import compute_classification_metrics, run_and_log_confidence
from torch.utils.data import WeightedRandomSampler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FKT-KE Federated Orchestration")
    parser.add_argument("--dataset_name", type=str, default="imdb", help="HF dataset name, e.g., 'imdb' or 'ag_news'")
    parser.add_argument("--num_clients", "-K", type=int, default=5, help="Total number of clients")
    parser.add_argument("--num_rounds", type=int, default=20, help="Total federated rounds")
    parser.add_argument("--editor_ckpt_path", type=str, default=None, help="Pretrained editor checkpoint path")
    parser.add_argument("--log_dir", type=str, default="runs", help="Root directory for TensorBoard logs")
    parser.add_argument("--alpha", type=float, default=config.NONIID_ALPHA, help="Dirichlet alpha for non-IID split")
    parser.add_argument("--enable_eval", action="store_true", help="Enable local eval and confidence logging")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use class-balanced WeightedRandomSampler during training")
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
    return compute_classification_metrics(predictions, labels)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _log_confidence_metrics(
    writer: SummaryWriter,
    tag_prefix: str,
    logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    global_step: int | None = None,
) -> None:
    # Convert logits to probabilities and confidences
    if logits.ndim == 1:
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.stack([1 - probs, probs], axis=1)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        probs = np.stack([1 - probs, probs], axis=1)
    else:
        probs = _softmax(logits, axis=1)

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    y_true = labels.astype(int)

    # Overall
    overall_acc = float((preds == y_true).mean())
    mean_conf = float(confidences.mean())
    writer.add_scalar(f"{tag_prefix}/confidence/overall_accuracy", overall_acc, global_step or 0)
    writer.add_scalar(f"{tag_prefix}/confidence/mean_confidence", mean_conf, global_step or 0)
    writer.add_histogram(f"{tag_prefix}/confidence/hist", confidences, global_step or 0)

    # Reliability diagram bins and Expected Calibration Error (ECE)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(num_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == 0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float((preds[mask] == y_true[mask]).mean())
        frac = float(mask.mean())
        ece += frac * abs(bin_acc - bin_conf)
        # Log per-bin scalars as a group
        writer.add_scalars(
            f"{tag_prefix}/confidence/reliability/bin_{i}",
            {"acc": bin_acc, "conf": bin_conf, "frac": frac},
            global_step or 0,
        )

    writer.add_scalar(f"{tag_prefix}/confidence/ECE", float(ece), global_step or 0)


def _finetune_client_with_lora(client: Client, output_dir: str, writer: SummaryWriter, enable_eval: bool, use_weighted_sampler: bool) -> None:
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
        evaluation_strategy="steps" if enable_eval else "no",
        eval_steps=config.TRAINING_DEFAULTS["logging_steps"],
        weight_decay=config.TRAINING_DEFAULTS.get("weight_decay", 0.0),
        warmup_ratio=config.TRAINING_DEFAULTS.get("warmup_ratio", 0.0),
        label_smoothing_factor=config.TRAINING_DEFAULTS.get("label_smoothing_factor", 0.0),
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

    # Build class-balanced sampler for training
    labels_np = np.array(train_ds["labels"]) if "labels" in train_ds.column_names else None
    train_sampler = None
    if use_weighted_sampler and labels_np is not None and labels_np.size > 0:
        unique_labs, counts = np.unique(labels_np, return_counts=True)
        # inverse frequency as weight
        freq = {int(l): float(c) for l, c in zip(unique_labs, counts)}
        weights = np.array([1.0 / (freq[int(l)] + 1e-12) for l in labels_np], dtype=np.float64)
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if enable_eval else None,
        data_collator=collator,
        compute_metrics=compute_metrics_fn if enable_eval else None,
        # pass sampler via data_collator is not supported; override get_train_dataloader instead
    )

    if train_sampler is not None:
        def _get_train_dataloader_override():
            from torch.utils.data import DataLoader
            return DataLoader(
                train_ds,
                batch_size=training_args.per_device_train_batch_size,
                sampler=train_sampler,
                collate_fn=collator,
            )
        trainer.get_train_dataloader = _get_train_dataloader_override

    # Attach TB loss logging
    trainer.add_callback(TBLossCallback(writer, tag_prefix=f"client_{client.client_id}"))
    trainer.train()

    # Confidence analysis on local eval set (only when enabled)
    if enable_eval:
        run_and_log_confidence(
            trainer=trainer,
            eval_dataset=eval_ds,
            writer=writer,
            tag_prefix=f"client_{client.client_id}",
            num_bins=10,
        )

        # Optional: Evaluate on global public dataset to avoid local skew
        # Note: `public_dataset` is not directly accessible here; compute in main and pass if needed.

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

    # Infer number of labels for the classification head
    labels_all = tokenized["labels"].numpy() if hasattr(tokenized["labels"], "numpy") else np.array(tokenized["labels"])
    unique_labels = np.unique(labels_all)
    num_labels = int(unique_labels.max() + 1) if unique_labels.min() >= 0 and unique_labels.max() + 1 == len(unique_labels) else int(len(unique_labels))

    # Build non-IID partitions
    print(f"[Info] Creating non-IID partitions: K={args.num_clients}, alpha={args.alpha}")
    private_datasets, public_dataset = create_non_iid_partitions(tokenized, args.num_clients, args.alpha)

    # Run-specific logging directory (unique per run)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}_ds-{args.dataset_name}_K{args.num_clients}_a{args.alpha}_eval{int(args.enable_eval)}"
    run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard writers under run_dir
    writer_global = SummaryWriter(log_dir=os.path.join(run_dir, "global"))
    client_writers = [SummaryWriter(log_dir=os.path.join(run_dir, f"client_{i}")) for i in range(args.num_clients)]

    # Log hyperparameters to file and TensorBoard
    hparams: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "alpha": args.alpha,
        "enable_eval": args.enable_eval,
        "seed": config.SEED,
        "model_name": config.MODEL_NAME,
        "lora_config": config.LORA_CONFIG,
        "training_defaults": config.TRAINING_DEFAULTS,
    }
    with open(os.path.join(run_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    writer_global.add_text("hparams", json.dumps(hparams, ensure_ascii=False, indent=2), global_step=0)

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
        clients.append(Client(
            client_id=client_id,
            private_dataset=private_datasets[client_id],
            device=device,
            tokenizer=tokenizer,
            num_labels=num_labels,
        ))

    # Pre-FL per-client LoRA warm-up
    for client in clients:
        output_dir = os.path.join(run_dir, "outputs", f"client_{client.client_id}_lora")
        os.makedirs(output_dir, exist_ok=True)
        _finetune_client_with_lora(
            client,
            output_dir=output_dir,
            writer=client_writers[client.client_id],
            enable_eval=args.enable_eval,
            use_weighted_sampler=args.use_weighted_sampler,
        )

    if args.enable_eval:
        # Evaluate each client model on the shared public dataset (balanced/global view)
        # Build eval dataset for public split using same tokenizer format (already tokenized earlier pipeline; need to map)
        def _public_tok_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
            text_col = "text"
            if text_col not in batch:
                for c in ["sentence", "content", "news", "review", "text"]:
                    if c in batch:
                        text_col = c
                        break
            toks = tokenizer(batch[text_col], truncation=True, max_length=256)
            if "labels" in batch:
                toks["labels"] = batch["labels"]
            elif "label" in batch:
                toks["labels"] = batch["label"]
            return toks

        public_tk = public_dataset.map(_public_tok_fn, batched=True, remove_columns=[c for c in public_dataset.column_names if c not in ["text", "label", "labels"]])
        public_tk.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        for client in clients:
            # temporary trainer to reuse eval loop on public set
            tmp_args = TrainingArguments(output_dir=os.path.join(run_dir, "tmp_public_eval"), per_device_eval_batch_size=config.TRAINING_DEFAULTS["per_device_train_batch_size"], report_to=[], remove_unused_columns=False)
            tmp_trainer = Trainer(model=client.model, args=tmp_args, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
            pred = tmp_trainer.predict(public_tk)
            metrics = compute_classification_metrics(pred.predictions if not isinstance(pred.predictions, (list, tuple)) else pred.predictions[0], pred.label_ids)
            writer_global.add_scalars(f"public_eval/client_{client.client_id}", metrics, global_step=0)

    print("[Info] LoRA fine-tuning completed for all clients. Ready for FL rounds.")

    # Placeholder for the main federated learning loop
    # for round_idx in range(args.num_rounds):
    #     ...


if __name__ == "__main__":
    main()


