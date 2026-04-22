import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np

# ==========================================
# 🛑 核心配置区
# ==========================================
DATA_DIR = "./dermnet_processed/shot_4_seed0" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RL_EPOCHS = 2000
LR = 5e-4

# 超参数：两个奖励的平衡
# Extrinsic (标签集准确率奖励) 的权重默认为 1.0
ALPHA_INTRINSIC = 0.1   # Intrinsic (无标签集自信奖励) 的权重，不能太大防止带偏

# ==========================================
# 1. 强化学习智能体 (Policy Network)
# ==========================================
class RLAgent(nn.Module):
    def __init__(self, input_dim=768, num_concepts=50):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
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
    print("🚀 正在加载 半监督 RL 数据集...")
    
    # 标签数据 (作为验证/准确率裁判 Anchor) - 92张
    labeled_data = torch.load(os.path.join(DATA_DIR, "val_data.pt"), map_location=DEVICE)
    X_labeled = labeled_data['clip_feats']
    y_labeled = labeled_data['true_labels']
    
    # 无标签数据 (作为浩瀚的探索池) - 3900+张
    # 注意：在半监督训练中，我们彻底假装看不见 y_unlabeled！
    unlabeled_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), map_location=DEVICE)
    X_unlabeled = unlabeled_data['clip_feats']
    y_ground_truth_for_final_test = unlabeled_data['true_labels'] # 仅用于最后看总成绩
    
    # 加载 Lasso 先验
    anchor_dict = torch.load(os.path.join(DATA_DIR, "anchor_dict.pt"), map_location=DEVICE)
    T_purified = anchor_dict['T_purified']
    M_cls = anchor_dict['M_cls']
    
    num_concepts = T_purified.shape[0]
    agent = RLAgent(input_dim=768, num_concepts=num_concepts).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    
    # 预计算全局 Concept Scores
    scores_labeled = 100.0 * (X_labeled @ T_purified.t())
    scores_unlabeled = 100.0 * (X_unlabeled @ T_purified.t())

    print("\n⚔️ 开始双轨 半监督强化学习 (Semi-Supervised RL)...")
    
    for epoch in range(RL_EPOCHS):
        optimizer.zero_grad()
        
        # --------------------------------------------------
        # 轨一：标签验证集计算 (Extrinsic Reward)
        # --------------------------------------------------
        # 智能体生成 Mask
        actions_lab, log_probs_lab, _, _ = agent(X_labeled)
        
        # 应用 Mask 并预测
        logits_lab = (scores_labeled * actions_lab) @ M_cls.t()
        preds_lab = logits_lab.argmax(dim=1)
        
        # 计算绝对准确率奖励 (+1 / -1)
        correct_lab = (preds_lab == y_labeled).float()
        rewards_lab = correct_lab * 1.0 - (1.0 - correct_lab) * 1.0
        # 奖励标准化 (Trick)
        rewards_lab = (rewards_lab - rewards_lab.mean()) / (rewards_lab.std() + 1e-8)
        
        # 标签集策略损失
        loss_labeled = -(log_probs_lab * rewards_lab.detach()).mean()

        # --------------------------------------------------
        # 轨二：无标签探索集计算 (Intrinsic Reward)
        # --------------------------------------------------
        # 每次随机抽一个 Batch 的无标签数据进行探索（防内存溢出）
        idx = torch.randperm(X_unlabeled.shape[0])[:256] 
        X_batch_unlabeled = X_unlabeled[idx]
        scores_batch_unlabeled = scores_unlabeled[idx]
        
        actions_unlab, log_probs_unlab, _, _ = agent(X_batch_unlabeled)
        
        # 应用 Mask 并计算 Softmax 概率
        logits_unlab = (scores_batch_unlabeled * actions_unlab) @ M_cls.t()
        probs_unlab = torch.softmax(logits_unlab, dim=1)
        
        # 计算预测分布的信息熵 (Entropy)
        # 熵越小 -> 越自信 -> 奖励应该越大
        pred_entropy = -(probs_unlab * torch.log(probs_unlab + 1e-8)).sum(dim=1)
        
        # 自信奖励 = 负的熵 (标准化)
        rewards_unlab = -pred_entropy
        rewards_unlab = (rewards_unlab - rewards_unlab.mean()) / (rewards_unlab.std() + 1e-8)
        
        # 无标签集策略损失
        loss_unlabeled = -(log_probs_unlab * rewards_unlab.detach()).mean()

        # --------------------------------------------------
        # 双轨合并，反向传播
        # --------------------------------------------------
        total_loss = loss_labeled + ALPHA_INTRINSIC * loss_unlabeled
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            acc_val = correct_lab.mean().item() * 100
            print(f"   [Epoch {epoch+1:3d}] Loss: {total_loss.item():.4f} | 标签集准确率 (Anchor): {acc_val:.2f}% | 无标签集熵: {pred_entropy.mean().item():.4f}")

    # --------------------------------------------------
    # 终极大考：验收所有无标签数据的真实准确率
    # --------------------------------------------------
    print("\n📊 训练完毕，正在开启“上帝视角”验证整体准确率...")
    agent.eval()
    with torch.no_grad():
        actions_final, _, _, _ = agent(X_unlabeled, deterministic=True)
        logits_final = (scores_unlabeled * actions_final) @ M_cls.t()
        preds_final = logits_final.argmax(dim=1)
        
        final_acc = (preds_final == y_ground_truth_for_final_test).float().mean().item()
        
        mask_sparsity = 1.0 - (actions_final.sum() / (actions_final.shape[0] * actions_final.shape[1])).item()

    print("=" * 60)
    print(f"👑 最终 半监督 RL 成绩单 (4-Shot + Unlabeled Pool):")
    print(f"   🏆 Concept-Guided Semi-RL 准确率:   {final_acc * 100:.2f}%")
    print(f"   💡 平均切断了 {mask_sparsity * 100:.1f}% 的实例级特征噪声！")
    print("=" * 60)