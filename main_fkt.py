from __future__ import annotations

import argparse
import os
import json
from typing import List, Optional, Dict, Any, Tuple

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime
import multiprocessing as mp

import config
from client import Client
from utils.data_utils import create_non_iid_partitions
from torch.utils.tensorboard import SummaryWriter
from utils.eval_utils import compute_classification_metrics, run_and_log_confidence
from torch.utils.data import WeightedRandomSampler
from editors.recipe.recipe import RECIPEConfig
from fkt_editor import FKTEditor
from torch.utils.data import DataLoader


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
    parser.add_argument("--parallel_warmup", action="store_true", help="Enable parallel LoRA warm-up across multiple GPUs")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids for parallel warm-up, e.g., '0,1,2'")
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


def _build_min_recipe_cfg(hidden: int = 768) -> RECIPEConfig:
    return RECIPEConfig(
        prompt_token_n=4,
        edit_model_name="gpt2",
        knowledge_rep_dim=256,
        knowl_rep_prot_token_n=1,
        model_hidden_size=hidden,
        begin_layer_path="transformer.h.0.attn",
        lm_head_path="lm_head",
        training=RECIPEConfig.TrainingConfig(
            krm_lr=1e-4,
            pt_lr=1e-4,
            relia_lambda=1.0,
            gen_lambda=1.0,
            loc_lambda=1.0,
            contra_lambda=1.0,
            query_knowledge_t=1.0,
            query_prototype_t=1.0,
            constra_hinge_scale=1.0,
            edit_hinge_scale=1.0,
        ),
    )


def _init_editor(device: torch.device) -> FKTEditor:
    lm_tok = AutoTokenizer.from_pretrained("gpt2")
    if lm_tok.pad_token_id is None:
        lm_tok.pad_token = lm_tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    cfg = _build_min_recipe_cfg(hidden=lm.config.n_embd)
    return FKTEditor(model=lm, tokenizer=lm_tok, config=cfg, device=str(device), ckpt_path=None)


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


@torch.no_grad()
def _predict_labels_and_confidences(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    logits = model(input_ids=batch["input_ids"].to(model.device), attention_mask=batch.get("attention_mask").to(model.device)).logits
    probs = torch.softmax(logits, dim=-1)
    confs, preds = torch.max(probs, dim=-1)
    return {"preds": preds.cpu().numpy(), "confs": confs.cpu().numpy(), "logits": logits.cpu().numpy()}


def _format_markdown_table(sample_texts: List[str], client_results: Dict[int, List[Tuple[int, float]]]) -> str:
    # Header
    headers = ["sample"] + [f"client_{cid}" for cid in sorted(client_results.keys())]
    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    # Rows
    num_rows = len(sample_texts)
    for i in range(num_rows):
        row = [sample_texts[i][:80].replace("|", "/")]  # truncate to 80 chars
        for cid in sorted(client_results.keys()):
            lab, cf = client_results[cid][i]
            row.append(f"{lab} ({cf:.2f})")
        md += "| " + " | ".join(row) + " |\n"
    return md


def _evaluate_client(client: Client, eval_ds: Dataset, writer: SummaryWriter, round_idx: int) -> float:
    client.model.eval()
    collator = DataCollatorWithPadding(tokenizer=client.tokenizer)
    dl = DataLoader(eval_ds, batch_size=client.model.config.to_dict().get("per_device_eval_batch_size", 32), shuffle=False, collate_fn=collator)
    correct = 0
    total = 0
    for batch in dl:
        # Decode texts for activation (if editor provided)
        if getattr(client, 'editor', None) is not None:
            texts = client.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            prompts = client.editor.dynamic_activation(texts)
            client.editor.adopted_prompts = [p.to(client.editor.device) for p in prompts]  # for hooked LMs
        out = client.model(input_ids=batch["input_ids"].to(client.device), attention_mask=batch.get("attention_mask").to(client.device))
        preds = out.logits.argmax(dim=-1).cpu()
        labels = batch["labels"].cpu()
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    acc = (correct / max(1, total))
    writer.add_scalar(f"client_{client.client_id}/accuracy", float(acc), round_idx)
    return float(acc)


def _parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    if not gpu_ids_str:
        return []
    parts = [p.strip() for p in gpu_ids_str.split(",") if p.strip() != ""]
    ids: List[int] = []
    for p in parts:
        try:
            ids.append(int(p))
        except ValueError:
            continue
    return ids


def _warmup_worker(
    client_id: int,
    private_dataset: Dataset,
    num_labels: int,
    model_name: str,
    run_dir: str,
    enable_eval: bool,
    use_weighted_sampler: bool,
    gpu_id: int,
) -> str:
    # Each process sees only one GPU to avoid intra-process DP and NCCL
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    client = Client(
        client_id=client_id,
        private_dataset=private_dataset,
        device=device,
        tokenizer=tokenizer,
        num_labels=num_labels,
    )

    # Independent writer per client
    writer = SummaryWriter(log_dir=os.path.join(run_dir, f"client_{client_id}"))
    output_dir = os.path.join(run_dir, "outputs", f"client_{client_id}_lora")
    os.makedirs(output_dir, exist_ok=True)

    _finetune_client_with_lora(
        client,
        output_dir=output_dir,
        writer=writer,
        enable_eval=enable_eval,
        use_weighted_sampler=use_weighted_sampler,
    )

    merged_dir = os.path.join(run_dir, "outputs", f"client_{client_id}_merged")
    os.makedirs(merged_dir, exist_ok=True)
    client.model.save_pretrained(merged_dir)
    return merged_dir


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

    # Pre-FL LoRA warm-up (sequential or parallel)
    if args.parallel_warmup and torch.cuda.is_available():
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        if len(gpu_ids) == 0:
            gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) == 0:
            print("[Warn] No GPUs available for parallel warm-up; falling back to sequential.")
        else:
            print(f"[Info] Parallel warm-up on GPUs: {gpu_ids}")
            with mp.get_context("spawn").Pool(processes=min(len(gpu_ids), args.num_clients)) as pool:
                jobs = []
                for client in clients:
                    assigned_gpu = gpu_ids[client.client_id % len(gpu_ids)]
                    jobs.append((
                        client.client_id,
                        client.private_data,
                        client.num_labels,
                        config.MODEL_NAME,
                        run_dir,
                        args.enable_eval,
                        args.use_weighted_sampler,
                        assigned_gpu,
                    ))
                merged_dirs: List[str] = pool.starmap(_warmup_worker, jobs)

            # Load merged weights back into existing client objects
            for client, mdir in zip(clients, merged_dirs):
                model = AutoModelForSequenceClassification.from_pretrained(mdir)
                client.model = model.to(device)

    if not args.parallel_warmup or not torch.cuda.is_available():
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

    # ------------------------- Federated Rounds ------------------------- #
    # Prepare a tokenized public set for batching (reuse public_tk if created; otherwise, build from tokenized train slices)
    public_pool = private_datasets[0]  # use format-compatible dataset; we'll sample indices from public_dataset later
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_base: Optional[Dataset] = None
    if "test" in raw:
        base_test: Dataset = raw["test"]
        def tok_fn_eval(batch: Dict[str, Any]) -> Dict[str, Any]:
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
        eval_base = base_test.map(tok_fn_eval, batched=True, remove_columns=[c for c in base_test.column_names if c not in ["text", "label", "labels"]])
        eval_base.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Initialize editors for clients
    editors: List[FKTEditor] = []
    for c in clients:
        ed = _init_editor(device)
        c.editor = ed
        editors.append(ed)

    for round_idx in range(args.num_rounds):
        print(f"[Round {round_idx}] Phase 1: inference on public batch")
        # Sample a small batch from public_dataset indices
        pub_size = len(public_dataset)
        if pub_size == 0:
            print("[Warn] Empty public dataset; skipping round.")
            continue
        batch_size = min(32, pub_size)
        sel = np.random.choice(pub_size, size=batch_size, replace=False)
        batch = public_dataset.select(sel)
        # Build tensors
        dl = DataLoader(batch, batch_size=batch_size, shuffle=False, collate_fn=collator)
        batch_tensors = next(iter(dl))
        # Decode sample texts for logging / conflict keys
        sample_texts = tokenizer.batch_decode(batch_tensors["input_ids"], skip_special_tokens=True)

        # Collect predictions per client
        all_predictions: Dict[int, List[Tuple[int, float]]] = {}
        for client in clients:
            preds = _predict_labels_and_confidences(client.model, batch_tensors)
            client_results = list(zip(preds["preds"].tolist(), preds["confs"].tolist()))
            all_predictions[client.client_id] = client_results

        # Log markdown table
        md = _format_markdown_table(sample_texts, all_predictions)
        writer_global.add_text(f"round_{round_idx}/public_predictions", md, global_step=round_idx)

        # Phase 2 & 3: conflict detection and ingestion
        print(f"[Round {round_idx}] Phase 2&3: conflict detection and ingestion")
        for receiver in clients:
            conflicts: Dict[Tuple[str, str], List[float]] = {}
            recv_res = all_predictions[receiver.client_id]
            for i, s_text in enumerate(sample_texts):
                y_k = recv_res[i][0]
                for other in clients:
                    if other.client_id == receiver.client_id:
                        continue
                    y_j, conf_j = all_predictions[other.client_id][i]
                    if int(y_j) != int(y_k):
                        key = (s_text, str(y_j))
                        conflicts.setdefault(key, []).append(float(conf_j))

            # Build candidates list
            candidates = []
            for (sample, label_text), conf_list in conflicts.items():
                candidates.append({
                    "sample": sample,
                    "label": label_text,
                    "confidence": float(np.mean(conf_list)),
                    "resonance": float(len(conf_list)),
                })
            if len(candidates) > 0:
                editors[receiver.client_id].ingest_knowledge(candidates)
            client_writers[receiver.client_id].add_scalar(
                f"client_{receiver.client_id}/new_knowledge_count", len(candidates), round_idx
            )

        # Phase 4: evaluation
        print(f"[Round {round_idx}] Phase 4: evaluation")
        eval_ds = eval_base if eval_base is not None else public_dataset
        for client in clients:
            acc = _evaluate_client(client, eval_ds, client_writers[client.client_id], round_idx)
            print(f"  Client {client.client_id} accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()


