import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import ElasticNet

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    medmnist = None

# ==========================================
# 🛑 统一配置中心 (Master Config)
# ==========================================
CONFIG = {
    # 1. 选择要跑的数据集: 'bloodmnist', 'dermnet', 'cub_200'
    "dataset_name": "bloodmnist",

    # 2. 填写数据集路径和概念字典路径
    "paths": {
        "bloodmnist": {
            "data_dir": "./data",                              
            "concept_json": "./concept/bloodmnist_concept.json",
            "save_root": "./bloodmnist_processed",
        },
        "dermnet": {
            "data_dir": "./data/dermnet/test",                
            "concept_json": "./concept/dermnet_concepts.json",
            "save_root": "./dermnet_processed",
        },
        "cub_200": {
            "data_dir": "./data/CUB_200_2011/images",           
            "concept_json": "./concept/cub_200_concepts.json",  
            "save_root": "./cub_200_processed",
        },
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14"

BATCH_SIZE = 16  # 为了应对 5-Crop TTA 带来的显存压力，适当调小 batch_size
SHOT_LIST = [1, 2, 4, 8, 16] # 用于计算全局采样总量的基数
NUM_SEEDS = 3

# 🔥 核心控制开关
USE_PURIFICATION = True     # False: 专家先验直通 | True: 数据驱动提纯
ELASTIC_ALPHA = 0.000001    # 正则化强度
ELASTIC_L1_RATIO = 0.5      # L1与L2的比例 (0.5表示各占一半，越接近1越像纯Lasso)

# ==========================================
# 📚 多视角文本模板 (Prompt Ensembling)
# ==========================================
CONCEPT_TEMPLATES = {
    "bloodmnist": [
       "A microscopic image showing {phrase}.",
        "A micrograph of a {phrase}.",
        "A blood smear image containing a {phrase}.",
        "A zoomed-in pathology photo of a {phrase}.",
        "A medical image of a {phrase}.",
        "A pathology slide showing a {phrase}.",
        "A Wright-Giemsa stained micrograph of a {phrase}.",
        "A close-up view of a {phrase} under a microscope."
    ],
    "dermnet": [
        "A close-up clinical photo showing {phrase}.",
        "A macroscopic view of {phrase} on skin.",
        "Dermatology image characterized by {phrase}.",
        "A medical photo highlighting {phrase}."
    ],
    "cub_200": [
        "A wildlife photo of a {phrase}.",
        "A bird featuring {phrase}.",
        "A nature photograph highlighting {phrase}."
    ],
}

# ==========================================
# 🛠️ 统一工具函数
# ==========================================

def clean_classname(name, dataset_name):
    """清洗类别名，统一为可读格式"""
    if dataset_name == 'cub_200':
        return name.split('.')[-1].replace('_', ' ').lower()
    return name

def get_random_sampled_indices(labels, total_labeled_samples, seed=42):
    """全局随机采样 (模拟真实的随机少样本场景，而非完美类平衡)"""
    np.random.seed(seed)
    total_samples = len(labels)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    
    # 从全体无标签池中随机抽取标定样本，其余作为无监督/测试集
    val_idx = all_indices[:total_labeled_samples]
    train_idx = all_indices[total_labeled_samples:]
    
    return np.array(train_idx), np.array(val_idx)

# ==========================================
# 1. 统一图像特征提取 / 加载 (带 TTA 支持)
# ==========================================
def extract_or_load_image_features(model, processor, d_name, data_dir, save_root):
    feat_path = os.path.join(save_root, "master_image_feats.pt")
    if os.path.exists(feat_path):
        print(f"📂 发现已缓存的全局特征，直接加载: {feat_path}")
        return torch.load(feat_path, map_location='cpu')

    print(f"📸 开始提取并缓存 [{d_name.upper()}] 所有图像的 CLIP 特征...")

    if d_name == 'bloodmnist':
        if medmnist is None:
            raise ImportError("请先安装 medmnist 库!")
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        transform = transforms.Compose([transforms.ToTensor()])
        # 1. 从 train 分割加载（用于 few-shot 切分，与 run_lasso / train_rl 一致）
        dataset = DataClass(split='train', transform=transform, download=False, size=224, root=data_dir)
        class_names = list(info['label'].values())
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_feats, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Extracting (Standard)"):
                inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(DEVICE)
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())
                all_labels.extend(labels[:, 0].numpy())

    else:   # ImageFolder 格式: dermnet, cub_200
        # 🚀 引入 TTA (FiveCrop 视角增强)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        class_names = [clean_classname(cls, d_name) for cls in dataset.classes]
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_feats, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Extracting (with 5-Crop TTA)"):
                bs, ncrops, c, h, w = images.size()
                
                # 折叠 batch 和 crops 维度，送入 CLIP
                images = images.view(-1, c, h, w).to(DEVICE)
                inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(DEVICE)
                
                feats = model.get_image_features(**inputs)
                
                # 展开回并在 crop 维度 (dim=1) 上求均值
                feats = feats.view(bs, ncrops, -1).mean(dim=1)
                
                # 重新 L2 归一化
                feats = feats / feats.norm(dim=-1, keepdim=True)
                
                all_feats.append(feats.cpu())
                all_labels.extend(labels.numpy())

    master_data = {
        'clip_feats': torch.cat(all_feats, dim=0),
        'true_labels': torch.tensor(all_labels, dtype=torch.long),
        'class_names': class_names
    }
    os.makedirs(save_root, exist_ok=True)
    torch.save(master_data, feat_path)
    print(f"   ✅ 特征已缓存: {feat_path} | 形状: {master_data['clip_feats'].shape}")
    return master_data

# ==========================================
# 2. 加载并构建初始文本字典 T_init (Prompt Ensembling)
# ==========================================
def build_T_init(json_path, templates, model, processor):
    print(f"\n📖 加载概念字典: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        concept_dict = json.load(f)

    all_phrases = []
    for cls, dims in concept_dict.items():
        for dim, phrases in dims.items():
            all_phrases.extend(phrases)

    unique_phrases = list(dict.fromkeys(all_phrases))
    print(f"   -> 提取出 {len(unique_phrases)} 个唯一的视觉短语！")

    print(f"   -> 正在使用 {len(templates)} 个 Prompt 模板进行特征集成 (Ensembling)...")
    all_text_feats = []
    
    with torch.no_grad():
        for i in range(0, len(unique_phrases), 64):
            batch_phrases = unique_phrases[i:i+64]
            batch_feats_ensembles = []
            
            for template_str in templates:
                texts = [template_str.format(phrase=p.lower()) for p in batch_phrases]
                inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                batch_feats_ensembles.append(feats)
            
            # [Num_templates, Batch_size, Dim] -> 求均值 -> [Batch_size, Dim]
            avg_feats = torch.stack(batch_feats_ensembles).mean(dim=0)
            avg_feats = avg_feats / avg_feats.norm(dim=-1, keepdim=True) 
            all_text_feats.append(avg_feats.cpu())

    T_init = torch.cat(all_text_feats, dim=0)
    return T_init, unique_phrases, list(concept_dict.keys()), concept_dict

# ==========================================
# 3A. 专家先验映射引擎 (不使用提纯时)
# ==========================================
def build_expert_M_cls(concept_dict, unique_phrases, class_names):
    num_classes = len(class_names)
    num_concepts = len(unique_phrases)
    M_cls = np.zeros((num_classes, num_concepts), dtype=np.float32)

    json_keys_lower = {k.lower(): k for k in concept_dict.keys()}

    for c_idx, c_name in enumerate(class_names):
        c_name_lower = c_name.lower()
        if c_name_lower in json_keys_lower:
            json_key = json_keys_lower[c_name_lower]

            phrases_for_class = []
            for dim, p_list in concept_dict[json_key].items():
                phrases_for_class.extend(p_list)

            for p in phrases_for_class:
                if p in unique_phrases:
                    p_idx = unique_phrases.index(p)
                    M_cls[c_idx, p_idx] = 1.0

            row_sum = np.sum(M_cls[c_idx])
            if row_sum > 0:
                M_cls[c_idx] = M_cls[c_idx] / row_sum
        else:
            print(f"⚠️ 警告: 类别 '{c_name}' 在 JSON 字典中未找到！")

    return torch.tensor(M_cls, dtype=torch.float32)

# ==========================================
# 3B. Elastic Net 核心提纯引擎
# ==========================================
def run_elastic_purification(val_data, T_init, num_classes, alpha, l1_ratio):
    img_feats = val_data['clip_feats'].numpy()
    labels = val_data['true_labels'].numpy()
    T_init_np = T_init.numpy().T

    M_cls_list = []
    global_active_indices = set()
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, max_iter=5000)

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            M_cls_list.append(np.zeros(T_init.shape[0]))
            continue

        y = np.mean(img_feats[idx], axis=0)
        model.fit(T_init_np, y)
        a = model.coef_

        global_active_indices.update(np.where(a > 0)[0].tolist())
        M_cls_list.append(a)

    M_cls_raw = np.stack(M_cls_list, axis=0)
    active_idx_list = sorted(list(global_active_indices))

    if len(active_idx_list) == 0:
        raise ValueError(f"🚨 提纯过度！Alpha={alpha} 导致概念全部清零！请调小 alpha。")

    M_cls_purified = M_cls_raw[:, active_idx_list]
    T_purified = T_init[active_idx_list, :]
    return torch.tensor(M_cls_purified, dtype=torch.float32), T_purified, active_idx_list

# ==========================================
# 🚀 主程序
# ==========================================
if __name__ == "__main__":
    d_name = CONFIG["dataset_name"]
    ds_cfg = CONFIG["paths"][d_name]
    data_dir = ds_cfg["data_dir"]
    json_path = ds_cfg["concept_json"]
    save_root = ds_cfg["save_root"]
    templates = CONCEPT_TEMPLATES[d_name]

    os.makedirs(save_root, exist_ok=True)

    print(f"🚀 初始化 {MODEL_NAME} on {DEVICE}...")
    print(f"📊 目标数据集: [{d_name.upper()}]")
    print(f"   数据目录: {data_dir}")
    print(f"   概念字典: {json_path}")
    print(f"   输出目录: {save_root}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # 1. 提取/加载全局图像特征
    master_data = extract_or_load_image_features(model, processor, d_name, data_dir, save_root)

    # 2. 构建文本字典 T_init
    T_init, unique_phrases, dict_class_names, concept_dict = build_T_init(
        json_path, templates, model, processor
    )

    num_classes = len(torch.unique(master_data['true_labels']))
    class_names = master_data.get('class_names', dict_class_names)

    if USE_PURIFICATION:
        print(f"\n✂️ 当前模式：[Elastic Net 数据驱动提纯] (Alpha={ELASTIC_ALPHA})")
    else:
        print(f"\n🧠 当前模式：[专家先验直通] (不经过特征丢弃)")

    for seed in range(NUM_SEEDS):
        print(f"\n🎲 ===== Seed {seed} =====")
        for shot in SHOT_LIST:
            shot_dir = os.path.join(save_root, f"shot_{shot}_seed{seed}")
            os.makedirs(shot_dir, exist_ok=True)

            total_labeled_samples = shot * num_classes

            train_idx, val_idx = get_random_sampled_indices(
                master_data['true_labels'].numpy(), total_labeled_samples, seed=42 + seed
            )
            print(f"   [Shot 基数 {shot}] 剩余无监督池(Train): {len(train_idx)} | 随机锚定样本(Val): {len(val_idx)}")

            train_data = {
                'clip_feats': master_data['clip_feats'][train_idx],
                'true_labels': master_data['true_labels'][train_idx]
            }
            val_data = {
                'clip_feats': master_data['clip_feats'][val_idx],
                'true_labels': master_data['true_labels'][val_idx]
            }

            torch.save(train_data, os.path.join(shot_dir, "train_data.pt"))
            torch.save(val_data, os.path.join(shot_dir, "val_data.pt"))

            if USE_PURIFICATION:
                M_cls, T_purified, active_indices = run_elastic_purification(
                    val_data, T_init, num_classes, ELASTIC_ALPHA, ELASTIC_L1_RATIO
                )
                print(f"      🌟 {len(unique_phrases)} 概念 -> {len(active_indices)} 锚点")
            else:
                active_indices = list(range(len(unique_phrases)))
                T_purified = T_init
                M_cls = build_expert_M_cls(concept_dict, unique_phrases, class_names)
                print(f"      🧠 使用全部 {len(active_indices)} 个概念")

            active_phrases = [unique_phrases[i] for i in active_indices]

            torch.save({
                'M_cls': M_cls,
                'T_purified': T_purified,
                'concept_indices': active_indices,
                'active_phrases': active_phrases
            }, os.path.join(shot_dir, "anchor_dict.pt"))

            # Markdown 报告
            report_path = os.path.join(shot_dir, f"elastic_report_shot{shot}_seed{seed}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# [{d_name.upper()}] Concept Routing Report (Base Shot {shot}, Seed {seed})\n\n")
                f.write(f"**Mode:** {'Elastic Net' if USE_PURIFICATION else 'Expert Prior'}\n")
                f.write(f"**Initial concepts:** {len(unique_phrases)}\n")
                f.write(f"**Survived concepts:** {len(active_indices)}\n\n")
                f.write("## Class-Concept Weight Distribution\n\n")

                M_cls_np = M_cls.numpy()
                for c in range(num_classes):
                    cname = class_names[c]
                    weights = M_cls_np[c]
                    active_c = np.where(weights > 0)[0]

                    if len(active_c) > 0:
                        f.write(f"### 【{cname}】({len(active_c)} activated)\n")
                        sorted_idx = active_c[np.argsort(-weights[active_c])]
                        for i in sorted_idx:
                            orig_i = active_indices[i]
                            phrase = unique_phrases[orig_i]
                            f.write(f"- **[{weights[i]:.4f}]** {phrase}\n")
                        f.write("\n")
                    else:
                        f.write(f"### 【{cname}】\n- No concept assigned!\n\n")

                print(f"      📁 报告已保存: {report_path}")

    print(f"\n🎉 全线竣工！数据和锚点已存入 {save_root}/ 文件夹。")