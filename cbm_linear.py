import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

# ==========================================
# 🛑 核心配置区
# ==========================================
JSON_PATH = "./concept/bloodmnist_concept.json"
FEAT_PATH = "./bloodmnist_processed/master_image_feats.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14"

SHOT_LIST = [1, 4]
NUM_SEEDS = 3       # 跑 3 次取平均，结果更具说服力
LASSO_ALPHA = 0.0001 # Lasso 惩罚力度

EPOCHS = 100
LR = 0.01

# ==========================================
# 工具函数
# ==========================================
def get_stratified_indices(labels, shots_per_class, seed):
    np.random.seed(seed)
    train_idx, test_idx = [], []
    num_classes = len(np.unique(labels))
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        num_to_take = min(shots_per_class, len(c_idx))
        train_idx.extend(c_idx[:num_to_take])     
        test_idx.extend(c_idx[num_to_take:])   
    return np.array(train_idx), np.array(test_idx)

# 构建未过滤概念字典
def build_T_init(json_path, model, processor):
    with open(json_path, 'r', encoding='utf-8') as f:
        concept_dict = json.load(f)
    all_phrases = []
    for cls, dims in concept_dict.items():
        for dim, phrases in dims.items():
            all_phrases.extend(phrases)
    unique_phrases = list(dict.fromkeys(all_phrases))
    
    texts = [f"A close-up clinical photo showing {phrase.lower()}." for phrase in unique_phrases]
    all_text_feats = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            batch_texts = texts[i:i+128]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True).to(DEVICE)
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_text_feats.append(feats.cpu())
    return torch.cat(all_text_feats, dim=0), unique_phrases

# 极简线性分类器
class LinearCBM(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.classifier(x)

# 训练与测试单个分类器的函数
def train_and_eval_linear(X_train, y_train, X_test, y_test, in_features, num_classes):
    model = LinearCBM(in_features, num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    
    # 训练
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    # 测试
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        preds = test_logits.argmax(dim=1).cpu().numpy()
        
    return accuracy_score(y_test.numpy(), preds)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print(f"🚀 初始化大模型提取概念特征...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    master_data = torch.load(FEAT_PATH, map_location='cpu')
    image_feats = master_data['clip_feats']
    labels = master_data['true_labels']
    num_classes = len(torch.unique(labels))

    T_init, unique_phrases = build_T_init(JSON_PATH, model, processor)
    total_concepts = T_init.shape[0]

    # 将所有图像投影到完整的概念空间 (N, 206)
    print("\n🧠 正在计算全局 Concept Scores (图像与所有概念的相似度)...")
    concept_scores = 100.0 * (image_feats @ T_init.t())

    print("\n" + "="*60)
    print(" ⚔️ 开始双轨消融实验 (Ablation Study) ⚔️")
    print("="*60)

    for shot in SHOT_LIST:
        acc_no_lasso = []
        acc_with_lasso = []
        
        print(f"\n🧪 测试组: [{shot}-Shot]")
        for seed in range(NUM_SEEDS):
            train_idx, test_idx = get_stratified_indices(labels.numpy(), shot, seed=42+seed)
            
            X_train_full = concept_scores[train_idx]
            y_train = labels[train_idx]
            X_test_full = concept_scores[test_idx]
            y_test = labels[test_idx]

            # --------------------------------------------------
            # 🔴 支路 A: 不加 Lasso (全量概念 200+ 维)
            # --------------------------------------------------
            acc_A = train_and_eval_linear(X_train_full, y_train, X_test_full, y_test, total_concepts, num_classes)
            acc_no_lasso.append(acc_A)

            # --------------------------------------------------
            # 🟢 支路 B: 加上 Lasso (动态降维)
            # --------------------------------------------------
            lasso = Lasso(alpha=LASSO_ALPHA, positive=True, max_iter=5000)
            active_indices = set()
            
            # 在训练集上寻找高纯度概念
            for c in range(num_classes):
                c_idx = np.where(y_train.numpy() == c)[0]
                if len(c_idx) > 0:
                    y_target = torch.mean(image_feats[train_idx][c_idx], dim=0).numpy()
                    lasso.fit(T_init.numpy().T, y_target)
                    active_indices.update(np.where(lasso.coef_ > 0)[0].tolist())
            
            active_indices = sorted(list(active_indices))
            num_purified = len(active_indices)
            
            if num_purified > 0:
                # 按照提纯后的概念切片
                X_train_sparse = X_train_full[:, active_indices]
                X_test_sparse = X_test_full[:, active_indices]
                acc_B = train_and_eval_linear(X_train_sparse, y_train, X_test_sparse, y_test, num_purified, num_classes)
            else:
                acc_B = 0.0 # 理论上不会发生，除非 Alpha 极大
                
            acc_with_lasso.append(acc_B)
            
            print(f"   [Seed {seed}] 🔴 无 Lasso (206维): {acc_A*100:.2f}% | 🟢 有 Lasso ({num_purified}维): {acc_B*100:.2f}%")

        print("-" * 60)
        print(f"   📊 [{shot}-Shot] 最终平均准确率:")
        print(f"      ❌ 无过滤直接分类 (Pure Linear CBM): {np.mean(acc_no_lasso)*100:.2f}%")
        print(f"      ✅ 结合 Lasso 降维 (Lasso + Linear): {np.mean(acc_with_lasso)*100:.2f}%")
        print("="*60)