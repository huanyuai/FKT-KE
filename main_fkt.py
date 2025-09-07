from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, TaskType, get_peft_model

import config
from client import Client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FKT-KE Federated Orchestration")
    parser.add_argument("--num_clients", "-K", type=int, default=5, help="Total number of clients")
    parser.add_argument("--num_rounds", "-R", type=int, default=20, help="Total federated rounds")
    parser.add_argument("--public_data_path", type=str, default=None, help="Path to public shared dataset")
    parser.add_argument(
        "--private_data_dir",
        type=str,
        default=None,
        help="Directory containing per-client private data (e.g., client_0.json or client_0/)",
    )
    parser.add_argument(
        "--editor_ckpt_path", type=str, default=None, help="Checkpoint path for the pretrained knowledge editor"
    )
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


def _resolve_client_private_path(base_dir: Optional[str], client_id: int) -> Optional[str]:
    if base_dir is None:
        return None
    candidates = [
        os.path.join(base_dir, f"client_{client_id}"),
        os.path.join(base_dir, f"client_{client_id}.json"),
        os.path.join(base_dir, f"client_{client_id}.jsonl"),
        os.path.join(base_dir, str(client_id)),
        os.path.join(base_dir, f"{client_id}.json"),
        os.path.join(base_dir, f"{client_id}.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # If none exists, return the base dir itself letting client fallback to synthetic data
    return base_dir


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=int(config.LORA_CONFIG.get("r", 8)),
        lora_alpha=int(config.LORA_CONFIG.get("lora_alpha", 16)),
        lora_dropout=float(config.LORA_CONFIG.get("lora_dropout", 0.1)),
        bias=str(config.LORA_CONFIG.get("bias", "none")),
        task_type=_task_type_from_string(config.LORA_CONFIG.get("task_type", "SEQ_CLS")),
    )


def _finetune_client_with_lora(client: Client, output_dir: str) -> None:
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
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=client.private_data,
        data_collator=collator,
    )

    trainer.train()

    # Merge LoRA weights back to the base model for later use in FL
    merged = peft_model.merge_and_unload()
    client.model = merged.to(client.device)


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)

    # Initialize clients
    clients: List[Client] = []
    for client_id in range(args.num_clients):
        private_path = _resolve_client_private_path(args.private_data_dir, client_id)
        clients.append(Client(client_id=client_id, private_data_path=private_path, device=device, tokenizer=tokenizer))

    # Pre-FL per-client LoRA warm-up
    for client in clients:
        output_dir = os.path.join("./outputs", f"client_{client.client_id}_lora")
        os.makedirs(output_dir, exist_ok=True)
        _finetune_client_with_lora(client, output_dir=output_dir)

    print("[Info] LoRA fine-tuning completed for all clients. Ready for FL rounds.")

    # Placeholder for the main federated learning loop
    # for round_idx in range(args.num_rounds):
    #     ...


if __name__ == "__main__":
    main()


