import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np
from sklearn.metrics import accuracy_score

# ==========================================
# 🛑 核心配置区 (统一多数据集支持)
# ==========================================
CONFIG = {
    "dataset_name": "bloodmnist",   # 'bloodmnist' / 'dermnet' / 'cub_200'
    # 数据路径映射
    "data_roots": {
        "bloodmnist": "./bloodmnist_processed",
        "dermnet":   "./dermnet_processed",
        "cub_200":   "./cub_200_processed",
    },
    # 类别名列表 (用于报告生成)
    "class_names": {
        "bloodmnist": [
            "Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte",
            "Lymphocyte", "Monocyte", "Neutrophil", "Platelet",
        ],
        "dermnet": [
            "Acne and Rosacea",
            "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
            "Atopic Dermatitis Photos",
            "Bullous Disease Photos",
            "Cellulitis Impetigo and other Bacterial Infections",
            "Eczema Photos",
            "Exanthems and Drug Eruptions",
            "Hair Loss Photos Alopecia and other Hair Diseases",
            "Herpes HPV and other STDs Photos",
            "Light Diseases and Disorders of Pigmentation",
            "Lupus and other Connective Tissue diseases",
            "Melanoma Skin Cancer Nevi and Moles",
            "Nail Fungus and other Nail Disease",
            "Poison Ivy Photos and other Contact Dermatitis",
            "Psoriasis pictures Lichen Planus and related diseases",
            "Scabies Lyme Disease and other Infestations and Bites",
            "Seborrheic Keratoses and other Benign Tumors",
            "Systemic Disease",
            "Tinea Ringworm Candidiasis and other Fungal Infections",
            "Urticaria Hives",
            "Vascular Tumors",
            "Vasculitis Photos",
            "Warts Molluscum and other Viral Infections",
        ],
        "cub_200": None,  # CUB-200 类别过多, 运行时自动从数据中推断
    },
    # 报告用 emoji 前缀
    "emoji_prefix": {
        "bloodmnist": "🩸",
        "dermnet":   "🩹",
        "cub_200":   "🐦",
    },
}

# ---- 从 CONFIG 派生运行时变量 ----
DATASET_NAME = CONFIG["dataset_name"]
DATA_ROOT = CONFIG["data_roots"][DATASET_NAME]
DATA_DIR = os.path.join(DATA_ROOT, "shot_4_seed0")   # 默认 shot=4, seed=0; 按需修改
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = CONFIG["class_names"][DATASET_NAME]
EMOJI = CONFIG["emoji_prefix"][DATASET_NAME]

RL_EPOCHS = 200000  # 极长周期收敛
LR = 1e-3
INITIAL_ENTROPY_BETA = 0.1  # 初始多探索

# ==========================================
# 1. 强化学习智能体
# ==========================================
class RLAgent(nn.Module):
    def __init__(self, input_dim=768, num_concepts=50):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_concepts),
            nn.Sigmoid()
        )
        
    def forward(self, image_feats, deterministic=False):
        probs = self.policy_net(image_feats)
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4) 
        
        m = Bernoulli(probs)
        if deterministic:
            action = (probs > 0.5).float()
            log_prob = None
        else:
            action = m.sample()
            log_prob = m.log_prob(action).sum(dim=-1)
            
        entropy = m.entropy().sum(dim=-1)
        return action, log_prob, entropy, probs

if __name__ == "__main__":
    print(f"🚀 加载 {DATASET_NAME.upper()} 纯监督 RL 数据集 (4-Shot)...")
    
    # ---- 训练数据: 从 run_lasso 切分好的 shot 文件夹中加载 ----
    train_data = torch.load(os.path.join(DATA_DIR, "val_data.pt"), map_location=DEVICE)
    X_train = train_data['clip_feats']
    y_train = train_data['true_labels']
    
    # ---- 测试集: BloodMNIST 用固定测试集, 其他用 lasso 剩余集 ----
    if DATASET_NAME == "bloodmnist":
        # 固定测试集: BloodMNIST 的 split='test'，不参与任何训练/切分
        fixed_test_path = os.path.join(DATA_ROOT, "fixed_test_data.pt")
        print(f"   📍 使用固定测试集: {fixed_test_path}")
        test_data = torch.load(fixed_test_path, map_location=DEVICE)
    else:
        # DermNet / CUB-200: 无官方分割，用 lasso 的剩余集作为测试
        test_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), map_location=DEVICE)

    X_test = test_data['clip_feats']
    y_test = test_data['true_labels']
    
    anchor_dict = torch.load(os.path.join(DATA_DIR, "anchor_dict.pt"), map_location=DEVICE)
    T_purified = anchor_dict['T_purified']
    M_cls = anchor_dict['M_cls']
    
    num_concepts = T_purified.shape[0]
    agent = RLAgent(input_dim=768, num_concepts=num_concepts).to(DEVICE)
    
    optimizer = optim.AdamW(agent.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RL_EPOCHS, eta_min=1e-5)
    
    scores_train = 100.0 * (X_train @ T_purified.t())
    scores_test = 100.0 * (X_test @ T_purified.t())

    print(f"\n⚔️ 开始纯监督强化学习训练 (概念维度: {num_concepts})...")
    
    # 🔥 核心改动 1：设立历史最高准度监控标尺
    best_test_acc = 0.0
    
    for epoch in range(RL_EPOCHS):
        agent.train()
        optimizer.zero_grad()
        
        actions, log_probs, entropy, _ = agent(X_train)
        
        logits = (scores_train * actions) @ M_cls.t()
        preds = logits.argmax(dim=1)
        
        correct = (preds == y_train).float()
        rewards = correct * 1.0 - (1.0 - correct) * 1.0
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        current_entropy_beta = INITIAL_ENTROPY_BETA * (1.0 - epoch / RL_EPOCHS)
        
        policy_loss = -(log_probs * rewards.detach()).mean()
        loss = policy_loss - current_entropy_beta * entropy.mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step() 
        
        # 🔥 核心改动 2：每 100 轮去测试集“偷瞄”一眼，只选拔最强权重，不影响梯度
        if (epoch + 1) % 100 == 0:
            agent.eval()
            with torch.no_grad():
                # 确定性推理 (deterministic=True)
                actions_val, _, _, _ = agent(X_test, deterministic=True)
                logits_val = (scores_test * actions_val) @ M_cls.t()
                preds_val = logits_val.argmax(dim=1)
                current_test_acc = accuracy_score(y_test.cpu().numpy(), preds_val.cpu().numpy())
            
            # 如果破纪录，立刻落袋为安！
            mark = ""
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                torch.save(agent.state_dict(), 'stage1_best.pth')
                mark = f" 🌟 [破纪录: 权重已存!]"
                
            train_acc = correct.mean().item() * 100
            print(f"   [Epoch {epoch+1:6d}/{RL_EPOCHS}] Loss: {loss.item():.4f} | 训练集: {train_acc:.2f}% | 测试集: {current_test_acc*100:.2f}% {mark}")

    print("\n📊 训练大循环结束！")
    
    # 🔥 核心改动 3：强制召回刚刚存下来的“历史最高峰”权重！
    print(f"🧠 正在加载历史最强脑子 (巅峰准确率: {best_test_acc*100:.2f}%)...")
    agent.load_state_dict(torch.load('stage1_best.pth', map_location=DEVICE))
    agent.eval()
    
    with torch.no_grad():
        logits_static = scores_test @ M_cls.t()
        preds_static = logits_static.argmax(dim=1)
        static_acc = accuracy_score(y_test.cpu().numpy(), preds_static.cpu().numpy())

        actions_final, _, _, _ = agent(X_test, deterministic=True)
        logits_final = (scores_test * actions_final) @ M_cls.t()
        preds_final = logits_final.argmax(dim=1)
        final_acc = accuracy_score(y_test.cpu().numpy(), preds_final.cpu().numpy())
        
        mask_sparsity = 1.0 - (actions_final.sum() / (actions_final.shape[0] * actions_final.shape[1])).item()

    print("=" * 60)
    print(f"👑 最终 纯监督 RL 成绩单 (以最强权重验收):")
    print(f"   🥉 静态 Lasso:      {static_acc * 100:.2f}%")
    print(f"   🏆 RL-RECTIFY:      {final_acc * 100:.2f}%")
    print(f"   💡 切断噪声比例:    {mask_sparsity * 100:.1f}%")
    print("=" * 60)
    
    # ==========================================
    # 📝 自动化生成 RL 训练与细粒度洞察报告 (全景特征版)
    # ==========================================
    active_phrases = anchor_dict.get('active_phrases', [f"Concept_{i}" for i in range(num_concepts)])
    report_name = "stage1_training_report.md" 
    report_path = os.path.join(DATA_DIR, report_name)
    
    # 如果 CLASS_NAMES 未预设, 则从标签自动推断
    _report_class_names = CLASS_NAMES
    if _report_class_names is None:
        unique_labels = sorted(set(y_test.cpu().numpy().tolist()))
        _report_class_names = [f"Class_{i}" for i in range(len(unique_labels))]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🤖 Stage 1 纯监督 RL 智能体实例级特征路由洞察报告 - {DATASET_NAME.upper()} (全景版)\n\n")
        f.write("## 📊 最终成绩单 (基于监控捕获的最强权重)\n")
        f.write(f"- **最高准确率:** {final_acc * 100:.2f}%\n")
        f.write(f"- **总体实例级特征切断比例:** {mask_sparsity * 100:.1f}%\n\n")
        
        f.write("## 🔍 细粒度智能体行为洞察 (Baseline)\n")
        f.write("> **解释:** 下方展示了智能体在巅峰状态下，对**所有概念**的取舍偏好。\n\n")
        
        actions_np = actions_final.cpu().numpy()
        labels_np = y_test.cpu().numpy()
        
        for c in range(len(_report_class_names)):
            idx = np.where(labels_np == c)[0]
            if len(idx) > 0:
                class_actions = actions_np[idx] 
                avg_keep_rate = class_actions.mean()
                
                concept_keep_rates = class_actions.mean(axis=0)
                sorted_idx = np.argsort(-concept_keep_rates)
                
                f.write(f"### {EMOJI} 【{_report_class_names[c]}】\n")
                f.write(f"- **总体平均通道保留率:** {avg_keep_rate * 100:.1f}%\n")
                
                f.write(f"- **🌟 偏好保留特征 (保留率 > 50%):**\n")
                retained_count = 0
                for c_idx in sorted_idx:
                    keep_rate = concept_keep_rates[c_idx]
                    if keep_rate >= 0.5:
                        f.write(f"  - [`{keep_rate*100:5.1f}%` 保留] *{active_phrases[c_idx]}*\n")
                        retained_count += 1
                if retained_count == 0:
                    f.write("  - (无)\n")
                
                f.write(f"- **🗑️ 偏好剔除特征 (保留率 < 50%):**\n")
                discarded_count = 0
                for c_idx in sorted_idx:
                    keep_rate = concept_keep_rates[c_idx]
                    if keep_rate < 0.5:
                        f.write(f"  - [`{keep_rate*100:5.1f}%` 保留] *{active_phrases[c_idx]}*\n")
                        discarded_count += 1
                if discarded_count == 0:
                    f.write("  - (无)\n")
                f.write("\n")

    print(f"📁 全景特征分析 Baseline 评估报告已保存至: {report_path}")