import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import os
import shutil
import numpy as np
import json
from tqdm import tqdm
from sklearn.linear_model import Lasso

# ================= ⚙️ 配置与 TODO =================
SHOT_LIST = [1, 2, 4] 
NUM_SEEDS = 5 
CLIP_MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_ROOT = "./bloodmnist_processed"        # 预处理数据保存路径
NUM_CLASSES = 8                            # BloodMNIST 有 8 个类别

# 🔴 TODO: 请将你用大模型生成的 JSON 字典路径填在这里
CONCEPT_JSON_PATH = "./concept/bloodminest_concept.json"

# 🔴 论文核心超参数：抠门老板的惩罚力度（可微调 0.01 ~ 0.1）
LASSO_ALPHA = 0.05  

# ================= 🛠️ 工具函数 =================
def get_stratified_indices(labels, shots_per_class, num_classes, seed=42):
    """
    分层采样：从全局数据中，为每个类抽取 shots_per_class 个样本作为 Val (金标准)，
    剩下的全部作为 Train (无标签探索池)。
    """
    np.random.seed(seed)
    val_idx = []
    train_idx = []
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        val_idx.extend(c_idx[:shots_per_class])
        train_idx.extend(c_idx[shots_per_class:])
    return np.array(train_idx), np.array(val_idx)

def save_subset(master_data, indices, save_path):
    subset = {
        'clip_feats': master_data['clip_feats'][indices],
        'true_labels': master_data['true_labels'][indices],
    }
    torch.save(subset, save_path)
    return subset

# ================= 🧠 核心模块一：加载与编码初始概念 =================
def build_T_init(json_path, model, processor, device):
    print(f"\n📖 [Phase 1a] 正在加载初始 LLM 概念字典: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 JSON 文件: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        concept_dict = json.load(f)

    all_phrases = []
    # 展平所有类别的所有维度特征
    for cls, dims in concept_dict.items():
        for dim, phrases in dims.items():
            all_phrases.extend(phrases)

    # 去重，保留唯一视觉概念
    unique_phrases = list(dict.fromkeys(all_phrases))
    print(f"   -> 从 JSON 中提取出 {len(unique_phrases)} 个唯一的视觉概念。")

    # 包装 Prompt Template
    texts = [f"A microscopic image showing {phrase.lower()}." for phrase in unique_phrases]

    print("   -> 正在使用 CLIP Text Encoder 提取 T_init 矩阵...")
    batch_size = 256
    all_text_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Concepts"):
            batch_texts = texts[i:i+batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True).to(device)
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_text_feats.append(feats.cpu())

    T_init = torch.cat(all_text_feats, dim=0) # Shape: [N_init, 768]
    print(f"   ✅ T_init 构建完成! 形状: {T_init.shape}")
    return T_init, unique_phrases

# ================= 🧠 核心模块二：Lasso 稀疏提纯 =================
def build_static_anchors_with_lasso(val_data, T_init, num_classes, alpha):
    print(f"   🎯 [Phase 1b] 开始执行 Lasso 零梯度提纯 (Alpha={alpha})...")
    img_feats = val_data['clip_feats'].numpy() # 1-shot 图片特征
    labels = val_data['true_labels'].numpy()

    N_init, feat_dim = T_init.shape
    # Lasso 方程 y = Xa 中的 X，这里 X 就是概念特征库的转置
    T_init_np = T_init.numpy().T 

    M_cls_list = []
    global_active_indices = set()

    # positive=True 强制概念权重非负（我们不想要“负相关”的概念，只要正面证实存在的概念）
    lasso = Lasso(alpha=alpha, positive=True, max_iter=5000) 

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            print(f"      ⚠️ 警告: 类别 {c} 没有金标准样本，使用全零兜底！")
            M_cls_list.append(np.zeros(N_init))
            continue

        # 如果是 2-shot 或 4-shot，取金标准图像的平均特征作为靶点 y
        y = np.mean(img_feats[idx], axis=0) 

        # 🔥 核心求解：在这毫秒级的一瞬间，几百个废话概念被压缩为 0！
        lasso.fit(T_init_np, y)
        a = lasso.coef_ 

        non_zeros = np.sum(a > 0)
        global_active_indices.update(np.where(a > 0)[0].tolist())
        M_cls_list.append(a)

    M_cls_raw = np.stack(M_cls_list, axis=0) # [num_classes, N_init]
    active_idx_list = sorted(list(global_active_indices))
    
    if len(active_idx_list) == 0:
        raise ValueError("🚨 提纯失败！所有概念都被清零了。请调小 LASSO_ALPHA (如改为 0.01)！")

    print(f"   🌟 全局提纯完成: 从 {N_init} 个初始概念中，共精炼出 {len(active_idx_list)} 个核心锚点概念！")

    # 提取非零子矩阵
    M_cls_purified = M_cls_raw[:, active_idx_list] # 静态类别原型 [num_classes, N_purified]
    T_purified = T_init[active_idx_list, :]        # 纯净概念字典 [N_purified, 768]
    
    return torch.tensor(M_cls_purified, dtype=torch.float32), T_purified, active_idx_list

# ================= 主程序 =================
if __name__ == "__main__":
    print("🚀 初始化模型与数据准备流水线...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    # ⚠️ 读取已提取的全局图像特征
    master_data_path = os.path.join(SAVE_ROOT, "master_image_feats.pt")
    if not os.path.exists(master_data_path):
        raise FileNotFoundError(f"请先运行全局特征提取，保存 {master_data_path}")
    
    master_data = torch.load(master_data_path, map_location='cpu')
    labels = master_data['true_labels'].numpy()
    num_classes = NUM_CLASSES

    # 1. 离线构建初始全局概念库 T_init
    T_init, phrase_list = build_T_init(CONCEPT_JSON_PATH, model, processor, DEVICE)

    print(f"\n✂️ 开始生成 Semi-Supervised 切分并执行 Lasso 提纯...")
    for seed in range(NUM_SEEDS):
        print(f"\n🎲 === Seed {seed} ===")
        for shot in SHOT_LIST:
            dir_name = f"shot_{shot}_seed{seed}"
            shot_dir = os.path.join(SAVE_ROOT, dir_name)
            
            if os.path.exists(shot_dir):
                shutil.rmtree(shot_dir)
            os.makedirs(shot_dir, exist_ok=True)
            
            # 1. 拆分数据
            train_idx, val_idx = get_stratified_indices(labels, shot, num_classes, seed=42 + seed)
            print(f"   [Shot {shot}] 无标签池(Train): {len(train_idx)} | 金标准(Val): {len(val_idx)}")
            
            # 2. 保存常规的图像特征数据
            save_subset(master_data, train_idx, os.path.join(shot_dir, "train_data.pt"))
            val_data = save_subset(master_data, val_idx, os.path.join(shot_dir, "val_data.pt"))
            
            # 3. 🔥 调用 Lasso，为这个特定的 1-shot 切分构建绝对静态灯塔！
            M_cls, T_purified, active_indices = build_static_anchors_with_lasso(
                val_data=val_data, 
                T_init=T_init, 
                num_classes=num_classes, 
                alpha=LASSO_ALPHA
            )
            
            # 4. 保存提纯后的灯塔字典！
            torch.save({
                'M_cls': M_cls,            # RL计算奖励的锚点 [num_classes, N_purified]
                'T_purified': T_purified,  # 送给RL观察的概念特征库 [N_purified, 768]
                'concept_indices': active_indices # 保存下来，方便以后知道到底是哪几句话被选中了
            }, os.path.join(shot_dir, "anchor_dict.pt"))
            
            print(f"   ✅ [Shot {shot}] Anchor 字典已固化并保存至: {dir_name}/anchor_dict.pt")
            
    print("\n🎉 第一阶段（Offline Initialization & Purification）全线竣工！")