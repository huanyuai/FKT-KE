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
        # 首先尝试用记忆库生成，这代表了客户端的“偏见”
        outs = self.editor.generate_with_memory([prompt], max_new_tokens=5, do_sample=False)
        pred_label = (outs[0].strip().split()[0]) if outs and outs[0].strip() else ""
        
        # 为了广播，我们还需要一个置信度。这里简化一下：
        # 如果是靠记忆生成的，置信度高；否则，靠原始模型。
        # 一个更严谨的方法是获取生成token的概率，但当前generate接口不直接返回。
        # 我们暂时用一个启发式方法。
        best_idx, score = self.editor.calculate_activation_potential(prompt)
        confidence = 0.5 + (score / 10.0) if best_idx != -1 else 0.5 # Heuristic confidence

        return {
            'prompt': prompt,
            'prediction': pred_label,
            'confidence': confidence,
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
    args = parser.parse_args()

    # --- 1. 加载共享模型与配置 ---
    model_path, config_path = get_model_editor_config_path(args.edit_model_name, 'fkt-ke')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device)
    config = FKTKEConfig.from_yaml(config_path)

    # --- 2. 初始化客户端 ---
    # --- 3. 加载并划分数据 ---
    full_train_data = load_and_format_agnews('train', args.max_train_samples)
    client_private_data = create_non_iid_partitions(full_train_data, args.num_clients)
    
    # --- 4. 使用 LoRA 训练并创建异构客户端 ---
    clients: List[FKTKEClient] = []
    for cid in range(args.num_clients):
        print(f"--- Training LoRA for Client {cid} ---")
        lora_model = train_client_lora(cid, base_model, tokenizer, client_private_data[cid], args.device)
        editor = FKTKE(lora_model, tokenizer, config, args.device)
        clients.append(FKTKEClient(cid, editor))

    public_data = load_and_format_agnews('train', args.max_pub_samples) # 公共数据可以是独立的或训练集的一部分
    test_data = load_and_format_agnews('test', args.max_test_samples)
    
    # --- 5. 运行模拟器 ---
    sim = FederatedSimulator(clients, public_data, test_data)
    sim.run(args.num_rounds)

