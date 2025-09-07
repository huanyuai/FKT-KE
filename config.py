"""
Global configuration for FKT-KE experiments.

This module centralizes immutable or rarely-changed configuration such as
the base model name and LoRA defaults used for quick client-side adaptation.
"""

from typing import Dict, Any


# Base model to use across clients. Must be compatible with
# transformers.AutoModelForSequenceClassification
MODEL_NAME: str = "roberta-base"


# Default LoRA settings for client-side quick adaptation
# These values are conservative to keep fine-tuning light-weight.
LORA_CONFIG: Dict[str, Any] = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    # One of: "SEQ_CLS", "CAUSAL_LM", "TOKEN_CLS", "SEQ_2_SEQ_LM"
    "task_type": "SEQ_CLS",
}


# Light training defaults for the quick client warm-up (before FL rounds)
TRAINING_DEFAULTS: Dict[str, Any] = {
    "per_device_train_batch_size": 8,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "logging_steps": 10,
    # Use max_steps to bound time regardless of dataset size
    "max_steps": 20,
    "gradient_accumulation_steps": 1,
}


