import os
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np
from sklearn.metrics import accuracy_score

# ==========================================
# 🛑 配置区
# ==========================================
DATA_DIR = "./bloodmnist_processed/shot_4_seed0" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HARD_THRESHOLD = 0.5  # 你提出的 50% 硬阈值

CLASS_NAMES = ["Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]

# ==========================================
# 1. 强化学习智能体 (仅用于提取阈值和做对比)
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
        
    def forward(self, image_feats, deterministic=True):
        probs = self.policy_net(image_feats)
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4) 
        if deterministic:
            action = (probs > 0.5).float()
        else:
            m = Bernoulli(probs)
            action = m.sample()
        return action

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("🚀 启动消融实验验证: 动态路由 vs. 硬阈值静态路由...")
    
    # 1. 加载数据与字典
    test_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), map_location=DEVICE)
    X_test = test_data['clip_feats']
    y_test = test_data['true_labels']
    
    anchor_dict = torch.load(os.path.join(DATA_DIR, "anchor_dict.pt"), map_location=DEVICE)
    T_purified = anchor_dict['T_purified']
    M_cls_original = anchor_dict['M_cls']
    num_concepts = T_purified.shape[0]
    num_classes = M_cls_original.shape[0]
    
    # 🔥 提取大白话文字，如果没提取到则用编号兜底
    active_phrases = anchor_dict.get('active_phrases', [f"Concept_{i}" for i in range(num_concepts)])
    
    scores_test = 100.0 * (X_test @ T_purified.t())
    
    # 2. 加载最强脑子
    agent = RLAgent(input_dim=768, num_concepts=num_concepts).to(DEVICE)
    try:
        agent.load_state_dict(torch.load('stage1_best.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("❌ 找不到 stage1_best.pth，请确保已运行训练脚本。")
        exit()
    agent.eval()

    with torch.no_grad():
        # ==========================================
        # 🟢 测试 1: 原版全保留 (Lasso 提纯后的基线)
        # ==========================================
        logits_all = scores_test @ M_cls_original.t()
        acc_all = accuracy_score(y_test.cpu().numpy(), logits_all.argmax(dim=1).cpu().numpy())
        
        # ==========================================
        # 🔵 测试 2: RL 实例级动态路由 (我们的主模型)
        # ==========================================
        actions_dynamic = agent(X_test, deterministic=True)
        logits_dynamic = (scores_test * actions_dynamic) @ M_cls_original.t()
        acc_dynamic = accuracy_score(y_test.cpu().numpy(), logits_dynamic.argmax(dim=1).cpu().numpy())
        
        # ==========================================
        # 🔴 测试 3: 硬阈值静态路由 (并可视化截取过程)
        # ==========================================
        print(f"\n" + "=" * 60)
        print(f"🔪 正在执行硬阈值截取 (Threshold = {HARD_THRESHOLD * 100}%)...")
        print("=" * 60)
        
        M_cls_static = M_cls_original.clone()
        actions_np = actions_dynamic.cpu().numpy()
        labels_np = y_test.cpu().numpy()
        
        # 准备写 Markdown 报告
        report_path = os.path.join(DATA_DIR, "ablation_static_mask_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 🔪 硬阈值静态面具生成报告 (Threshold = {HARD_THRESHOLD*100}%)\n\n")
            f.write("> **说明:** 这是基于 RL 智能体动态期望生成的静态面具。保留率大于阈值的特征被固化为 1 (强制看)，小于阈值的被固化为 0 (强制蒙上)。\n\n")
        
            for c in range(num_classes):
                idx = np.where(labels_np == c)[0]
                if len(idx) > 0:
                    class_actions = actions_np[idx]
                    concept_keep_rates = class_actions.mean(axis=0) 
                    
                    kept_features = []
                    dropped_features = []
                    
                    # 遍历每一个概念，执行裁决
                    for k in range(num_concepts):
                        keep_rate = concept_keep_rates[k]
                        phrase = active_phrases[k]
                        
                        # 这个概念本来在 M_cls 里是不是分配给这个类的？
                        # 如果原来 M_cls 里权重就是 0，那不用管它
                        if M_cls_original[c, k].item() > 0: 
                            if keep_rate >= HARD_THRESHOLD:
                                kept_features.append((keep_rate, phrase))
                            else:
                                dropped_features.append((keep_rate, phrase))
                                M_cls_static[c, k] = 0.0 # 🔪 核心动作：彻底抹杀！
                    
                    # 排序以便打印
                    kept_features.sort(key=lambda x: x[0], reverse=True)
                    dropped_features.sort(key=lambda x: x[0], reverse=True)
                    
                    # -------- 控制台可视化打印 (仅抽样打印几个类防刷屏) --------
                    if c < 3 or c == num_classes - 1:
                        print(f"🩸 【{CLASS_NAMES[c]}】 面具生成:")
                        print(f"   ✔️ 固化保留 ({len(kept_features)} 个):")
                        for rate, phrase in kept_features[:2]: # 打印前两个
                            print(f"      [留存率 {rate*100:5.1f}% > 50%] -> 锁定为 1: {phrase}")
                        if len(kept_features) > 2: print("      ...")
                            
                        print(f"   ❌ 无情抛弃 ({len(dropped_features)} 个):")
                        for rate, phrase in dropped_features[:2]:
                            print(f"      [留存率 {rate*100:5.1f}% < 50%] -> 清零为 0: {phrase}")
                        if len(dropped_features) > 2: print("      ...")
                        print("-" * 50)
                    
                    # -------- 写入 Markdown 报告 (完整版) --------
                    f.write(f"### 🩸 【{CLASS_NAMES[c]}】\n")
                    f.write(f"- **✔️ 固化保留特征 (强制为 1):**\n")
                    for rate, phrase in kept_features:
                        f.write(f"  - [`{rate*100:5.1f}%`] *{phrase}*\n")
                    if not kept_features: f.write("  - (无)\n")
                        
                    f.write(f"- **❌ 强制剥夺特征 (清零为 0):**\n")
                    for rate, phrase in dropped_features:
                        f.write(f"  - [`{rate*100:5.1f}%`] *{phrase}*\n")
                    if not dropped_features: f.write("  - (无)\n")
                    f.write("\n")

        print(f"📁 详细的硬阈值截取白皮书已保存至: {report_path}")
                        
        # 3. 使用这副“死板的面具”进行无 RL 推理
        logits_static = scores_test @ M_cls_static.t()
        acc_static = accuracy_score(y_test.cpu().numpy(), logits_static.argmax(dim=1).cpu().numpy())

    # ==========================================
    # 📊 打印华丽的消融实验对比表格
    # ==========================================
    print("\n" + "=" * 65)
    print("🔥 消融实验结果 (Ablation Study on Routing Strategy) 🔥")
    print("=" * 65)
    print(f"{'路由策略 (Routing Strategy)':<30} | {'粒度':<10} | {'准确率 (Accuracy)'}")
    print("-" * 65)
    print(f"{'1. 全特征直通 (All Features)':<30} | {'无':<10} | {acc_all * 100:.2f}%")
    print(f"{'2. 硬阈值掩码 (Hard Threshold >50%)':<30} | {'类级别':<10} | {acc_static * 100:.2f}%")
    print(f"{'3. RL-Rectify (Ours: Dynamic)':<30} | {'实例级':<10} | {acc_dynamic * 100:.2f}%")
    print("=" * 65)
    
    if acc_dynamic > acc_static:
        print("\n💡 结论印证: 实例级动态路由完美击败了类级别硬阈值！")
        print("因为硬阈值一刀切地剥夺了模型在面对『困难/边缘变异样本』时所需的边缘特征补偿能力。")
    else:
        print("\n🤔 意外发现: 硬阈值竟然赢了？这说明数据集极其规范，没有太多变异的 Hard Samples。")