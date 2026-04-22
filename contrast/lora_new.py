import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    medmnist = None

# ==========================================
# 🛑 统一配置中心 (Master Config)
# ==========================================
CONFIG = {
    "dataset_name": "ham10000",  
    "shots_per_class": 4,

    "paths": {
        "bloodmnist": "./data",                               
        "neu_cls": "./data/NEU-CLS-ImageFolder",              
        "cub_200": "./data/CUB_200_2011/images",              
        "dermnet": "./data/dermnet/test",
        "ham10000": "./data/HAM10000/organizedimages",          # 7类皮肤病变 (10015张)
    }
}

BATCH_SIZE = 32 # 论文指定 Batch Size 为 32 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14" # 论文主要消融实验基座为 ViT-B/16 [cite: 204]

# 📝 固定测试集划分参数
FIXED_TEST_RATIO = 0.2      # 20% 作为固定测试集
SPLIT_SEED = 0              # 固定种子，确保跨脚本可复现

# 📝 CLIP 图像归一化 (对齐原版 datasets/utils.py)
CLIP_NORMALIZE = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

# 📝 论文指定：放弃 Prompt Ensemble，对所有数据集仅使用单一极简模板 
PROMPT_TEMPLATE = "a photo of a {}."

# ==========================================
# 🛠️ 统一数据加载与清洗模块 (保持不变)
# ==========================================
def clean_classname(name, dataset_name):
    if dataset_name == 'neu_cls':
        return name.replace('_', ' ').lower()
    elif dataset_name == 'cub_200':
        return name.split('.')[-1].replace('_', ' ').lower()
    return name

def split_fixed_test(labels, test_ratio=0.2, seed=0):
    """分层固定划分：按类别比例切分出固定测试集，返回 (train_indices, fixed_test_indices)
    
    与 BloodMNIST 的 split='test' 逻辑一致，保证跨脚本可复现。
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_test = max(1, int(len(c_idx) * test_ratio))  # 每类至少1张
        train_idx.extend(c_idx[:-n_test])
        test_idx.extend(c_idx[-n_test:])
    return np.array(train_idx), np.array(test_idx)

def load_or_split_fixed_test(dataset_name, data_dir):
    """加载或划分固定测试集缓存文件
    
    对于 ImageFolder 数据集（HAM10000/DermNet/CUB-200），首次运行时做 80/20 分层划分并缓存，
    之后直接读取缓存。确保 run_lasso / new_lasso / lora / stage* 使用完全相同的测试集。
    """
    # 缓存路径: 存放在数据目录旁的 _splits/ 文件夹下
    cache_dir = os.path.join(os.path.dirname(data_dir.rstrip('/\\')), "_splits")
    cache_path = os.path.join(cache_dir, f"{dataset_name}_fixed_test.pt")
    
    if os.path.exists(cache_path):
        print(f"   📍 发现固定测试集缓存: {cache_path}")
        data = torch.load(cache_path, map_location='cpu')
        return data['test_indices'], data['train_pool_indices'], data.get('class_names', None)
    
    print(f"   📐 首次运行，正在执行 {int(FIXED_TEST_RATIO*100)}% 固定测试集分层划分...")
    
    # 加载数据集以获取标签（transform 用轻量的 ToTensor 即可）
    if dataset_name == 'bloodmnist':
        if medmnist is None:
            raise ImportError("请先安装 medmnist 库！")
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset_tmp = DataClass(split='test', transform=transforms.ToTensor(), download=False, size=224, root=data_dir)
        class_names = list(info['label'].values())
        labels = np.array([label[0] for _, label in dataset_tmp])
    else:
        tmp_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset_tmp = datasets.ImageFolder(root=data_dir, transform=tmp_transform)
        class_names = [clean_classname(cls, dataset_name) for cls in dataset_tmp.classes]
        labels = np.array(dataset_tmp.targets)
    
    train_pool_idx, fixed_test_idx = split_fixed_test(labels, FIXED_TEST_RATIO, SPLIT_SEED)
    
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({
        'test_indices': fixed_test_idx,
        'train_pool_indices': train_pool_idx,
        'class_names': class_names,
        '_source': f'{dataset_name}_stratified_{FIXED_TEST_RATIO}_seed{SPLIT_SEED}'
    }, cache_path)
    
    n_train = len(train_pool_idx)
    n_test = len(fixed_test_idx)
    print(f"   ✅ 划分完成 | 训练池: {n_train} 张 | 固定测试集: {n_test} 张 | 缓存: {cache_path}")
    
    # 打印每类分布
    for c in range(len(class_names)):
        c_total = (labels == c).sum()
        c_test = (labels[fixed_test_idx] == c).sum()
        print(f"      {class_names[c]}: 总计={c_total}, 测试={c_test} ({100*c_test/c_total:.0f}%)")
    
    return fixed_test_idx, train_pool_idx, class_names

def load_dataset(dataset_name, data_dir):
    """加载训练集 (BloodMNIST 用 split='train', 其他保持原逻辑)"""
    print(f"\n📂 正在加载数据集: [{dataset_name.upper()}] ...")
    if dataset_name == 'bloodmnist':
        if medmnist is None:
            raise ImportError("请先安装 medmnist 库！")
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        transform = transforms.Compose([transforms.ToTensor(), CLIP_NORMALIZE])
        # 🔥 改为从 train 分割加载（用于 few-shot 切分）
        dataset = DataClass(split='train', transform=transform, download=False, size=224, root=data_dir)
        class_names = list(info['label'].values())
        labels = np.array([label[0] for _, label in dataset]) 
    elif dataset_name in ['neu_cls', 'cub_200', 'dermnet', 'ham10000']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            CLIP_NORMALIZE
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        class_names = [clean_classname(cls, dataset_name) for cls in dataset.classes]
        labels = np.array(dataset.targets)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")
    return dataset, class_names, labels


def load_fixed_test_dataset(dataset_name, data_dir):
    """加载固定测试集 (仅 BloodMNIST 使用 split='test')"""
    print(f"\n📂 正在加载固定测试集: [{dataset_name.upper()}] ...")
    if dataset_name == 'bloodmnist':
        if medmnist is None:
            raise ImportError("请先安装 medmnist 库！")
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        transform = transforms.Compose([transforms.ToTensor(), CLIP_NORMALIZE])
        dataset = DataClass(split='test', transform=transform, download=False, size=224, root=data_dir)
        class_names = list(info['label'].values())
        labels = np.array([label[0] for _, label in dataset])
    else:
        return None, None, None  # 非血细胞数据集无固定测试集
    return dataset, class_names, labels

# ==========================================
# 🚀 主程序
# ==========================================
if __name__ == "__main__":
    d_name = CONFIG["dataset_name"]
    k_shots = CONFIG["shots_per_class"]
    data_path = CONFIG["paths"][d_name]

    print(f"🚀 初始化 {MODEL_NAME} on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 1. 统一加载与切分（固定测试集划分）
    dataset, class_names, all_labels = load_dataset(d_name, data_path)
    
    # 1a. 固定测试集划分（所有 ImageFolder 数据集统一走此路径）
    if d_name != 'bloodmnist':
        fixed_test_indices, train_pool_indices, _ = load_or_split_fixed_test(d_name, data_path)
        # 从训练池中做 few-shot 切分
        pool_labels = all_labels[train_pool_indices]
        print(f"\n✂️ 正在从训练池({len(train_pool_indices)}张)中切分 {k_shots}-shot 标定样本...")
        train_indices, val_pool_indices = [], []
        for c in range(len(class_names)):
            c_pool_idx = train_pool_indices[np.where(pool_labels == c)[0]]
            # 在训练池内按类 shuffle 后采样
            perm = np.random.permutation(len(c_pool_idx))
            num_take = min(k_shots, len(perm))
            train_indices.extend(c_pool_idx[perm[:num_take]])
            val_pool_indices.extend(c_pool_idx[perm[num_take:]])
        print(f"   训练样本: {len(train_indices)} | 无标签池: {len(val_pool_indices)} | 固定测试: {len(fixed_test_indices)}")
    else:
        # BloodMNIST: 使用官方 test 分割作为固定测试集
        _, _, _ = load_or_split_fixed_test(d_name, data_path)  # 确保缓存一致
        print(f"\n✂️ 正在切分 {k_shots}-shot 训练集...")
        train_indices, test_indices = [], []
        for c in range(len(class_names)):
            c_idx = np.where(all_labels == c)[0]
            np.random.shuffle(c_idx)
            num_to_take = min(k_shots, len(c_idx))
            train_indices.extend(c_idx[:num_to_take])
            test_indices.extend(c_idx[num_to_take:])
        
    # 2. 准备固定的文本输入 (不需要提取特征，因为特征在训练中会变)
    texts = [PROMPT_TEMPLATE.format(cls) for cls in class_names]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)

    # 3. 注入 LoRA (严格遵循论文参数)
    print("\n💉 正在向 Vision 和 Text 双端注入 LoRA 权重...")
    lora_config = LoraConfig(
        r=2,                         # 论文明确要求 r=2 
        lora_alpha=1,                # 对齐原版 lora_main: alpha=1
        target_modules=["q_proj", "k_proj", "v_proj"], # 注入到 Query, Key, Value 
        lora_dropout=0.25,           # 论文指定 dropout 为 0.25 
        bias="none"
    )
    # peft 默认会匹配所有包含 q/k/v_proj 的层，即 Vision 和 Text 编码器都会被注入 
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)
    model.print_trainable_parameters()

    # 4. 设置优化器与学习率调度器
    # 论文指定学习率 2e-4，并使用余弦退火调度器 (对齐原始实现)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    
    train_subset = Subset(dataset, train_indices)
    train_bs = min(BATCH_SIZE, len(train_indices))
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True, num_workers=0)

    # 🔥 核心修正：总迭代次数 = n_iters × shots（与原版一致）
    N_ITERS = 500
    total_iters = N_ITERS * k_shots
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)

    print(f"\n🔥 开始严格复现版 {k_shots}-shot 训练...", flush=True)
    print(f"   总迭代次数: {total_iters} (n_iters={N_ITERS} × shots={k_shots})", flush=True)
    
    # ---- 混合精度加速 (与原版一致) ----
    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    model.train()
    count_iters = 0
    
    # 🔥 对齐原版：用 while 循环按 iteration 控制，而非 epoch
    while count_iters < total_iters:
        for images, targets in tqdm(train_loader, desc=f"[Iter {count_iters}/{total_iters}]", leave=False):
            images = images.to(DEVICE)
            targets = targets.view(-1).long().to(DEVICE)
            
            optimizer.zero_grad()
            
            # 图像已在 transform 中完成 Resize+ToTensor+CLIPNormalize，直接包装为 pixel_values
            image_inputs = {"pixel_values": images.to(DEVICE)}
            
            # 双端 LoRA 都在更新，必须在循环内同时计算图文特征
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = model.get_image_features(**image_inputs)
                    text_features = model.get_text_features(**text_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logit_scale = model.logit_scale.exp()
                    logits = logit_scale * image_features @ text_features.t()
                    loss = criterion(logits, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            
            # 🔥 每个迭代 step 一次（与原版一致）
            scheduler.step()
            count_iters += 1
            
            if count_iters >= total_iters:
                break
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"   [Iter {count_iters}/{total_iters}] LR: {current_lr:.6f}", flush=True)

    # 5. 测试评估
    print("\n📊 正在测试集上评估...")
    model.eval()
    
    # 评估前提取一次最新的文本特征即可
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ---- 选择测试数据源（统一使用固定测试集） ----
    if d_name == 'bloodmnist':
        print("   📍 使用固定测试集 (split='test')...")
        test_dataset, _, _ = load_fixed_test_dataset(d_name, data_path)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        # HAM10000 / DermNet / CUB-200 等: 使用分层划分的固定测试集
        test_subset = Subset(dataset, fixed_test_indices)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"   📍 使用固定测试集 ({len(fixed_test_indices)}张, {int(FIXED_TEST_RATIO*100)}%分层划分)...")
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            # 图像已在 transform 中完成 Resize+ToTensor+CLIPNormalize，直接包装为 pixel_values
            image_inputs = {"pixel_values": images}
            
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.view(-1).numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"\n>>> [{d_name.upper()}] {k_shots}-Shot CLIP-LORA 复现版准确率: {acc * 100:.2f}% <<<")