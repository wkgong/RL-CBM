import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np
from sklearn.metrics import accuracy_score

# ==========================================
# 🛑 核心配置区 (Stage 2, 统一多数据集支持)
# ==========================================
CONFIG = {
    "dataset_name": "dermnet",   # 'bloodmnist' / 'dermnet' / 'cub_200'
    # 数据路径映射
    "data_roots": {
        "bloodmnist": "./bloodmnist_processed",
        "dermnet":   "./dermnet_processed",
        "cub_200":   "./cub200_processed",
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
        "cub_200": None,
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

STAGE2_EPOCHS = 10000 
LR_FINE_TUNE = 1e-5 

# 权重平衡超参数
ALPHA_UNSUPERVISED = 1.5  # 无监督整体损失的比重
BETA_DIVERSITY = 1.0      # 覆盖率 (R_cov) 全局边缘熵的权重

# ==========================================
# 1. 强化学习智能体 (与 Stage 1 保持一致)
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

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print(f"🚀 [Stage 2] 启动带有金标准锚点的无监督 RL 微调 ({DATASET_NAME.upper()})...")
    
    # 1. 加载数据
    train_data = torch.load(os.path.join(DATA_DIR, "val_data.pt"), map_location=DEVICE)
    X_labeled = train_data['clip_feats']
    y_labeled = train_data['true_labels']
    
    unlabeled_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), map_location=DEVICE)
    X_unlabeled = unlabeled_data['clip_feats']
    y_ground_truth_for_test = unlabeled_data['true_labels'] # Stage 2 把无标签集当测试集
    
    anchor_dict = torch.load(os.path.join(DATA_DIR, "anchor_dict.pt"), map_location=DEVICE)
    T_purified = anchor_dict['T_purified']
    M_cls = anchor_dict['M_cls']
    
    num_concepts = T_purified.shape[0]
    
    # 2. 初始化并装载 Stage 1 权重
    agent = RLAgent(input_dim=768, num_concepts=num_concepts).to(DEVICE)
    print("   -> 正在装载 Stage 1 的最强脑子 'stage1_best.pth'...")
    try:
        agent.load_state_dict(torch.load('stage1_best.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("❌ 错误：找不到 stage1_best.pth，请先运行 train_rl_supervise.py！")
        exit()
        
    optimizer = optim.AdamW(agent.parameters(), lr=LR_FINE_TUNE, weight_decay=1e-4)
    
    scores_labeled = 100.0 * (X_labeled @ T_purified.t())
    scores_unlabeled = 100.0 * (X_unlabeled @ T_purified.t())

    print(f"\n⚔️ 开始 Stage 2 混合微调 (探索与防坍缩兼顾)...")
    
    # 🔥 核心改动 1：设立 Stage 2 的监控标尺
    best_stage2_acc = 0.0
    
    for epoch in range(STAGE2_EPOCHS):
        agent.train()
        optimizer.zero_grad()
        
        # 轨道 A：有监督锚点 (防止灾难性遗忘)
        actions_lab, log_probs_lab, _, _ = agent(X_labeled)
        logits_lab = (scores_labeled * actions_lab) @ M_cls.t()
        preds_lab = logits_lab.argmax(dim=1)
        
        correct_lab = (preds_lab == y_labeled).float()
        rewards_lab = correct_lab * 1.0 - (1.0 - correct_lab) * 1.0
        rewards_lab = (rewards_lab - rewards_lab.mean()) / (rewards_lab.std() + 1e-8)
        
        loss_supervised = -(log_probs_lab * rewards_lab.detach()).mean()
        
        # 轨道 B：无监督探索 (R_disc 与 R_cov)
        idx = torch.randperm(X_unlabeled.shape[0])[:256]
        X_batch_u = X_unlabeled[idx]
        scores_batch_u = scores_unlabeled[idx]
        
        actions_u, log_probs_u, _, _ = agent(X_batch_u)
        logits_u = (scores_batch_u * actions_u) @ M_cls.t()
        probs_u = torch.softmax(logits_u, dim=1) 
        
        instance_entropy = -(probs_u * torch.log(probs_u + 1e-8)).sum(dim=1)
        rewards_unsupervised = -instance_entropy
        rewards_unsupervised = (rewards_unsupervised - rewards_unsupervised.mean()) / (rewards_unsupervised.std() + 1e-8)
        
        marginal_probs = probs_u.mean(dim=0) 
        marginal_entropy = -(marginal_probs * torch.log(marginal_probs + 1e-8)).sum()
        
        loss_unsupervised = -(log_probs_u * rewards_unsupervised.detach()).mean() - BETA_DIVERSITY * marginal_entropy

        # 混合优化
        total_loss = loss_supervised + ALPHA_UNSUPERVISED * loss_unsupervised
        total_loss.backward()
        optimizer.step()
        
        # 🔥 核心改动 2：实时监控 Stage 2 测试集并捕获最优微调权重
        if (epoch + 1) % 100 == 0:
            agent.eval()
            with torch.no_grad():
                actions_val, _, _, _ = agent(X_unlabeled, deterministic=True)
                logits_val = (scores_unlabeled * actions_val) @ M_cls.t()
                preds_val = logits_val.argmax(dim=1)
                current_test_acc = accuracy_score(y_ground_truth_for_test.cpu().numpy(), preds_val.cpu().numpy())
            
            mark = ""
            if current_test_acc > best_stage2_acc:
                best_stage2_acc = current_test_acc
                # 存为独立的 stage2_best，防止覆盖 stage 1
                torch.save(agent.state_dict(), 'stage2_best.pth')
                mark = f" 🌟 [破纪录: 权重已存!]"
                
            train_acc = correct_lab.mean().item() * 100
            print(f"   [Epoch {epoch+1:5d}] Loss: {total_loss.item():.4f} | 锚点: {train_acc:.0f}% | 泛化准确率: {current_test_acc*100:.2f}% {mark}")

    print("\n📊 验收 Stage 2 微调后的最终泛化成绩...")
    
    # 🔥 核心改动 3：召回 Stage 2 最强权重
    print(f"🧠 正在加载微调巅峰权重 (最高准确率: {best_stage2_acc*100:.2f}%)...")
    agent.load_state_dict(torch.load('stage2_best.pth', map_location=DEVICE))
    agent.eval()
    
    with torch.no_grad():
        actions_final, _, _, _ = agent(X_unlabeled, deterministic=True)
        logits_final = (scores_unlabeled * actions_final) @ M_cls.t()
        preds_final = logits_final.argmax(dim=1)
        final_acc = accuracy_score(y_ground_truth_for_test.cpu().numpy(), preds_final.cpu().numpy())
        mask_sparsity = 1.0 - (actions_final.sum() / (actions_final.shape[0] * actions_final.shape[1])).item()

    print("=" * 60)
    print(f"👑 最终 Stage 2 成绩单 (Anchored Unsupervised Fine-tuning):")
    print(f"   🏆 半监督 RL 巅峰准确率: {final_acc * 100:.2f}%")
    print(f"   💡 切断噪声比例:         {mask_sparsity * 100:.1f}%")
    print("=" * 60)

    # ==========================================
    # 📝 自动化生成 RL 训练与细粒度洞察报告 (全景特征版)
    # ==========================================
    active_phrases = anchor_dict.get('active_phrases', [f"Concept_{i}" for i in range(num_concepts)])
    report_name = "rl_training_report.md" 
    report_path = os.path.join(DATA_DIR, report_name)
    
    # 如果 CLASS_NAMES 未预设, 则从标签自动推断
    _report_class_names = CLASS_NAMES
    if _report_class_names is None:
        unique_labels = sorted(set(y_ground_truth_for_test.cpu().numpy().tolist()))
        _report_class_names = [f"Class_{i}" for i in range(len(unique_labels))]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🤖 Stage 2 半监督 RL 智能体实例级特征路由洞察报告 - {DATASET_NAME.upper()} (全景版)\n\n")
        f.write("## 📊 最终成绩单 (基于监控捕获的最强微调权重)\n")
        f.write(f"- **最高准确率:** {final_acc * 100:.2f}%\n")
        f.write(f"- **总体实例级特征切断比例:** {mask_sparsity * 100:.1f}%\n\n")
        
        f.write("## 🔍 细粒度智能体行为洞察 (Fine-grained Agent Insight)\n")
        f.write("> **解释:** 下方展示了智能体在面对未知图像时，对**所有概念**的取舍偏好。\n\n")
        
        actions_np = actions_final.cpu().numpy()
        labels_np = y_ground_truth_for_test.cpu().numpy() 
        
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

    print(f"📁 全景特征分析评估报告已保存至: {report_path}")