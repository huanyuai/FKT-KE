"""
FKT-KE Simulation - Step 1/5

Goal:
- 设置项目环境并为联邦文本分类准备非IID数据集。

包含内容：
- 1) 导入与全局配置
- 2) 数据加载与非IID划分（Dirichlet）
- 3) 验证块（打印各客户端类别分布）
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


# === 1. Imports and Global Configuration ===
# 超参数
NUM_CLIENTS: int = 5
BATCH_SIZE: int = 32
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME: str = "distilbert-base-uncased"
DATASET_NAME: str = "ag_news"
NUM_CLASSES: int = 4  # For AG News


def _set_global_seeds(seed: int = 42) -> None:
    """设置随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detect_text_field(dataset: Dataset) -> str:
    """自动检测文本字段名称。

    优先顺序："text" → "content" → "description" → ("title" + "description") → 任意字符串字段
    如果存在 "title" 与 "description"，且没有单独的 "text" 字段，可在上游合并后再使用。
    这里返回一个单字段名；如需合并，可在上游先构造。
    """
    candidate_fields = ["text", "content", "description"]
    for field in candidate_fields:
        if field in dataset.column_names:
            return field

    # 如果只有 title/description 之类，尝试选择第一个字符串字段
    for name in dataset.column_names:
        # 简单启发：若某一列首元素为 str，则认为是文本列
        try:
            if isinstance(dataset[0][name], str):
                return name
        except Exception:
            continue

    # 兜底：返回第一列名
    return dataset.column_names[0]


def prepare_datasets(
    num_clients: int,
    dataset_name: str,
    model_name: str = MODEL_NAME,
    alpha: float = 0.5,
    max_length: int = 128,
    seed: int = 42,
) -> Tuple[List[Dataset], Dataset, Dataset, AutoTokenizer]:
    """加载数据集并进行非IID划分。

    步骤：
    - 加载 HuggingFace 数据集（例如 ag_news）。
    - 将原始 train 切分：80% 为私有客户端数据池，20% 为公共池（本函数中不返回公共池）。
    - 将原始 test 切分：50% 作为共享公共数据集 D_pub，50% 作为全局测试集。
    - 使用 Dirichlet 分布（alpha）对私有数据池做非IID划分，得到每个客户端的私有数据集。
    - 对客户端数据、D_pub、全局测试集进行分词（padding/truncation）。

    返回：
    - client_datasets: 每个客户端的私有 Dataset 列表
    - public_dataset: 共享公共数据集（来自原始 test 的 50%）
    - global_test_dataset: 全局测试集（来自原始 test 的 50%）
    - tokenizer: 分词器实例
    """
    _set_global_seeds(seed)

    # 加载数据集
    raw_datasets: DatasetDict = load_dataset(dataset_name)

    # 训练集：80% 用于私有划分；20% 作为公共池（此处不返回，但保留切分以符合需求）
    train_split = raw_datasets["train"].train_test_split(test_size=0.2, seed=seed)
    private_pool: Dataset = train_split["train"]  # 80%
    train_public_pool: Dataset = train_split["test"]  # 20%（当前不返回）

    # 测试集：50% 为 D_pub，50% 为全局测试集
    test_split = raw_datasets["test"].train_test_split(test_size=0.5, seed=seed)
    public_dataset_raw: Dataset = test_split["test"]
    global_test_raw: Dataset = test_split["train"]

    # 识别文本字段
    text_field: str = _detect_text_field(private_pool)

    # 构建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # === 非IID划分：对 private_pool 使用 Dirichlet(alpha) ===
    labels = np.array(private_pool["label"])  # 假设存在 label 字段
    num_classes = int(labels.max()) + 1 if "label" in private_pool.column_names else NUM_CLASSES
    if num_classes != NUM_CLASSES:
        # 若数据集标签数与全局常量不符，以真实标签数为准
        num_classes = NUM_CLASSES if NUM_CLASSES > 0 else num_classes

    # 为每个客户端收集样本索引
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in range(NUM_CLASSES):
        class_mask = labels == class_id
        class_indices = np.where(class_mask)[0]
        np.random.shuffle(class_indices)

        # 采样 Dirichlet 比例并根据该比例做多项式分配
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        counts = np.random.multinomial(len(class_indices), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                selected = class_indices[start : start + count].tolist()
                client_indices[client_id].extend(selected)
                start += count

    # 构造每个客户端的子数据集（未分词 → 先选择再分词，避免冗余 token 存储）
    client_datasets_raw: List[Dataset] = [
        private_pool.select(sorted(indices)) if len(indices) > 0 else private_pool.select([])
        for indices in client_indices
    ]

    # 分词前确定需要移除的原始列（保留 label）
    def columns_to_remove(ds: Dataset) -> List[str]:
        keep_cols = {"label"}
        return [c for c in ds.column_names if c not in keep_cols]

    # 对客户端数据进行分词
    tokenized_client_datasets: List[Dataset] = []
    for ds in client_datasets_raw:
        if len(ds) == 0:
            tokenized_client_datasets.append(ds)  # 空集直接返回
            continue
        tds = ds.map(tokenize_function, batched=True, remove_columns=columns_to_remove(ds))
        tokenized_client_datasets.append(tds)

    # 对公共数据与全局测试集进行分词
    public_dataset = public_dataset_raw.map(
        tokenize_function, batched=True, remove_columns=columns_to_remove(public_dataset_raw)
    )
    global_test_dataset = global_test_raw.map(
        tokenize_function, batched=True, remove_columns=columns_to_remove(global_test_raw)
    )

    return tokenized_client_datasets, public_dataset, global_test_dataset, tokenizer


# === 3. Verification Block ===
if __name__ == "__main__":
    print("Preparing datasets with non-IID partitioning...")
    clients_data, d_pub, d_test, tok = prepare_datasets(
        num_clients=NUM_CLIENTS,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        alpha=0.5,
        max_length=128,
        seed=42,
    )

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Public Dataset Size (D_pub): {len(d_pub)} | Global Test Size: {len(d_test)}")

    # 计算并打印每个客户端的类别分布
    for cid, ds in enumerate(clients_data):
        if len(ds) == 0:
            dist = [0] * NUM_CLASSES
        else:
            labels = np.array(ds["label"]) if "label" in ds.column_names else np.array([])
            if labels.size == 0:
                dist = [0] * NUM_CLASSES
            else:
                counts = np.bincount(labels, minlength=NUM_CLASSES)
                dist = counts.tolist()
        print(f"Client {cid} Class Distribution: {dist}")

    print("Verification complete. Non-IID distributions should show skewed class counts per client.")


