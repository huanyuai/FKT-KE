#%%
from typing import List, Dict, Any
import random
from peft import get_peft_model, LoraConfig, TaskType

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from editors.fkt_ke import FKTKE


def load_and_format_agnews(split: str, max_samples: int = None) -> List[Dict[str, str]]:
    """加载 AG News 并格式化为分类 prompt 列表。"""
    print(f"Loading and formatting AG News '{split}' split...")
    ds = load_dataset('ag_news', split=split)
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    samples: List[Dict[str, str]] = []
    total = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in range(total):
        sample = ds[i]
        prompt_text = ' '.join(sample['text'].split()[:50])
        prompt = f"Article: {prompt_text}\nCategory:"
        label_text = label_map[int(sample['label'])]
        samples.append({'prompt': prompt, 'label': label_text, 'label_id': sample['label']})
    print("Done.")
    return samples


def create_non_iid_partitions(dataset: List[Dict[str, Any]], num_clients: int) -> List[List[Dict[str, Any]]]:
    """
    创建一个 Non-IID 数据划分，模拟客户端的兴趣偏好。
    这里使用一个简单的类别偏斜方法。
    """
    print(f"Creating Non-IID partitions for {num_clients} clients...")
    # 按标签对数据进行排序
    sorted_data = sorted(dataset, key=lambda x: x['label_id'])
    # 将数据分成 N 个分片，N=类别数*2 (这里简化为 num_clients*2)
    num_shards = num_clients * 2
    shard_size = len(sorted_data) // num_shards
    shards = [sorted_data[i*shard_size : (i+1)*shard_size] for i in range(num_shards)]
    
    client_partitions = [[] for _ in range(num_clients)]
    # 每个客户端分配 2 个分片
    for i in range(num_clients):
        # 确保每个客户端拿到的数据类别有差异
        shard_idx1 = i
        shard_idx2 = (i + num_clients) % num_shards
        client_partitions[i].extend(shards[shard_idx1])
        client_partitions[i].extend(shards[shard_idx2])
        print(f"Client {i} has {len(client_partitions[i])} samples.")
    
    return client_partitions


def train_client_lora(client_id, base_model, tokenizer, private_data, device='cuda'):
    """为某个客户端在其私有数据上进行简化的 LoRA 微调，返回 LoRA 模型。"""
    # 1) 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"],
    )

    # 2) 模型准备
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model.to(device)

    # 3) 数据格式化
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [f"{s['prompt']} {s['label']}" for s in private_data]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # 4) 训练逻辑（精简版）
    from torch.optim import AdamW
    optimizer = AdamW(lora_model.parameters(), lr=1e-4)
    lora_model.train()

    epochs = 1
    batch_size = 8
    num_samples = input_ids.size(0)
    num_steps = (num_samples + batch_size - 1) // batch_size
    for epoch in range(epochs):
        pbar = tqdm(range(num_steps), desc=f"Client {client_id} LoRA Epoch {epoch+1}")
        for step in pbar:
            start = step * batch_size
            end = min(start + batch_size, num_samples)
            batch_input_ids = input_ids[start:end].to(device)
            batch_attn = attention_mask[start:end].to(device)
            batch_labels = labels[start:end].to(device)

            outputs = lora_model(input_ids=batch_input_ids,
                                 attention_mask=batch_attn,
                                 labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": float(loss.item())})

    return lora_model


class FKTKEClient:
    def __init__(self, client_id: int, editor: FKTKE) -> None:
        self.client_id = client_id
        self.editor = editor

    def populate_initial_knowledge(self, private_data: List[Dict[str, Any]]):
        """用私有数据填充初始知识库，模拟本地学习过程。"""
        print(f"Client {self.client_id} is populating its initial knowledge base from {len(private_data)} private samples...")
        requests = []
        metadatas = []
        for sample in tqdm(private_data, desc=f"Client {self.client_id} Learning"):
            requests.append({'prompt': sample['prompt'], 'target_new': sample['label']})
            # 来自本地数据的知识被认为是高置信度的
            metadatas.append({'E': 0.99, 'R': 1})
        self.editor.edit_batch(requests, metadatas=metadatas)

    @torch.no_grad()
    def broadcast_prediction(self, prompt: str) -> Dict[str, Any]:
        """
        执行内部仲裁，决定最终的预测和置信度。
        """
        editor = self.editor
        tok = editor.tokenizer
        
        # --- 顾问1：获取 LoRA 模型的直接预测 ---
        # 确保不使用记忆库，直接调用原始模型
        editor.active_prompt = None 
        inputs = tok(prompt, return_tensors='pt').to(editor.device)
        outputs = editor.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        lora_conf, lora_pred_id = torch.max(probs, dim=-1)
        lora_pred_label = tok.decode(lora_pred_id, skip_special_tokens=True).strip()
        
        # --- 顧問2：获取记忆库的建议 ---
        best_mem_idx, mem_score = editor.calculate_activation_potential(prompt)
        
        # --- 仲裁决策 ---
        # 我们需要一个标准来比较 lora_conf (0-1) 和 mem_score (无界)。
        # 策略：如果记忆库中存在一个足够强的信号，并且该信号超过了LoRA模型自身的置信度，则采纳记忆。
        # 这是一个启发式规则，可以进一步调优。
        # 将 mem_score 归一化到一个可比较的范围，例如通过一个缩放因子。
        normalized_mem_score = mem_score / 10.0 # 假设10分是一个很强的激活信号
        
        final_prediction = lora_pred_label
        final_confidence = lora_conf.item()

        if best_mem_idx != -1 and normalized_mem_score > lora_conf.item():
            # 使用记忆生成
            outs = editor.generate_with_memory([prompt], max_new_tokens=5, do_sample=False)
            mem_pred_label = (outs[0].strip().split()[0]) if outs and outs[0].strip() else ""
            
            if mem_pred_label:
                final_prediction = mem_pred_label
                final_confidence = normalized_mem_score # 使用归一化的记忆分数作为置信度
        
        return {
            'prompt': prompt,
            'prediction': final_prediction,
            'confidence': final_confidence,
        }


class FederatedSimulator:
    def __init__(self,
                 clients: List[FKTKEClient],
                 public_dataset: List[Dict[str, str]],
                 test_dataset: List[Dict[str, str]]) -> None:
        self.clients = clients
        self.public_dataset = public_dataset
        self.test_dataset = test_dataset

    def run_round(self) -> None:
        if not self.public_dataset:
            print("Public dataset is empty. Skipping round.")
            return
        
        batch = random.sample(self.public_dataset, k=min(5, len(self.public_dataset)))

        all_broadcasts_for_round: List[Dict[str, Any]] = []
        for sample in tqdm(batch, desc="Broadcasting Predictions"):
            for client in self.clients:
                pkt = client.broadcast_prediction(sample['prompt'])
                pkt['client_id'] = client.client_id
                all_broadcasts_for_round.append(pkt)

        for client in self.clients:
            knowledge_to_add: List[Dict[str, str]] = []
            metadata_to_add: List[Dict[str, float]] = []
            for sample in batch:
                related = [b for b in all_broadcasts_for_round if b['prompt'] == sample['prompt']]
                mine_list = [b for b in related if b.get('client_id') == client.client_id]
                if not mine_list: continue
                
                my_pred = mine_list[0]['prediction']
                others = [b for b in related if b.get('client_id') != client.client_id]
                
                resonance_map: Dict[str, List[Dict]] = {}
                for b in others:
                    pred = b['prediction']
                    if not pred or pred.strip() == my_pred.strip():
                        continue
                    if pred not in resonance_map:
                        resonance_map[pred] = []
                    resonance_map[pred].append(b)

                for pred_label, broadcasts in resonance_map.items():
                    if not broadcasts: continue
                    
                    rep = max(broadcasts, key=lambda x: x.get('confidence', 0.0))
                    request = {'prompt': sample['prompt'], 'target_new': pred_label}
                    metadata = {
                        'E': float(rep.get('confidence', 0.0)),
                        'R': float(len(broadcasts)), # Resonance is the count
                    }
                    knowledge_to_add.append(request)
                    metadata_to_add.append(metadata)

            if knowledge_to_add:
                print(f"Client {client.client_id} is consolidating {len(knowledge_to_add)} new knowledge pieces.")
                client.editor.edit_batch(knowledge_to_add, metadatas=metadata_to_add)

    def evaluate(self, round_num: int) -> None:
        print(f"\n===== [Evaluate] Round {round_num} =====")
        total_acc = 0
        for client in self.clients:
            correct = 0
            total = 0
            for sample in tqdm(self.test_dataset, desc=f"Evaluating Client {client.client_id}"):
                outs = client.editor.generate_with_memory([sample['prompt']], max_new_tokens=5, do_sample=False)
                text = outs[0] if outs else ''
                pred_list = text.strip().split()
                pred_label = pred_list[0] if pred_list else ''
                
                if pred_label.lower().strip() == sample['label'].lower().strip():
                    correct += 1
                total += 1

            acc = (correct / total) * 100 if total > 0 else 0.0
            total_acc += acc
            kb_size = len(getattr(client.editor, 'memory_keys', []))
            print(f"Client {client.client_id} -> Accuracy: {acc:.2f}%, Knowledge Base Size: {kb_size}")
        
        avg_acc = total_acc / len(self.clients) if self.clients else 0
        print(f"----- Average Accuracy for Round {round_num}: {avg_acc:.2f}% -----")


    def run(self, num_rounds: int) -> None:
        # 初始评估（此时客户端已有本地知识）
        self.evaluate(round_num=0)
        for i in range(num_rounds):
            print(f"\n{'='*20} [Round] {i+1}/{num_rounds} {'='*20}")
            self.run_round()
            self.evaluate(round_num=i+1)

if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils.utils import get_model_editor_config_path
    from editors.fkt_ke import FKTKE, FKTKEConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--edit_model_name', type=str, default='gpt2-xl')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_train_samples', type=int, default=1000, help="Total samples for creating Non-IID partitions.")
    parser.add_argument('--max_pub_samples', type=int, default=100)
    parser.add_argument('--max_test_samples', type=int, default=200)
    parser.add_argument('--krm_ckpt_path', type=str, default=None, help='Path to pretrained KnowledgeRepModel ckpt')
    parser.add_argument('--prompt_transformer_ckpt_path', type=str, default=None, help='Path to pretrained PromptTransformer ckpt')
    args = parser.parse_args()

    # --- 1a. 解析设备参数 ---
    if args.device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        resolved_device = args.device
    print(f"Resolved device for training and operations: {resolved_device}")

    # --- 1. 加载共享模型与配置 ---
    model_path, config_path = get_model_editor_config_path(args.edit_model_name, 'fkt-ke')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading base model onto single device: {resolved_device} for LoRA training...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path).to(resolved_device)
    config = FKTKEConfig.from_yaml(config_path)
    # 覆盖可选的任务对齐权重路径（如果从命令行提供）
    if args.krm_ckpt_path is not None:
        config.krm_ckpt_path = args.krm_ckpt_path
    if args.prompt_transformer_ckpt_path is not None:
        config.prompt_transformer_ckpt_path = args.prompt_transformer_ckpt_path

    # --- 2. 初始化客户端 ---
    # --- 3. 加载并划分数据 ---
    full_train_data = load_and_format_agnews('train', args.max_train_samples)
    client_private_data = create_non_iid_partitions(full_train_data, args.num_clients)
    
    # --- 4. 使用 LoRA 训练并创建异构客户端 ---
    clients: List[FKTKEClient] = []
    for cid in range(args.num_clients):
        print(f"--- Training LoRA for Client {cid} ---")
        lora_model = train_client_lora(cid, base_model, tokenizer, client_private_data[cid], resolved_device)
        editor = FKTKE(lora_model, tokenizer, config, resolved_device)
        clients.append(FKTKEClient(cid, editor))

    public_data = load_and_format_agnews('train', args.max_pub_samples) # 公共数据可以是独立的或训练集的一部分
    test_data = load_and_format_agnews('test', args.max_test_samples)
    
    # --- 5. 运行模拟器 ---
    sim = FederatedSimulator(clients, public_data, test_data)
    sim.run(args.num_rounds)

