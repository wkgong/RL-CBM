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
    "dataset_name": "bloodmnist",  
    "shots_per_class": 4,       
    "paths": {
        "bloodmnist": "./data",                              
        "neu_cls": "./data/NEU-CLS-ImageFolder",             
        "cub_200": "./data/CUB_200_2011/images",              
        "dermnet": "./data/dermnet/test",
    }
}

BATCH_SIZE = 32 # 论文指定 Batch Size 为 32 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14" # 论文主要消融实验基座为 ViT-B/16 [cite: 204]

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

def load_dataset(dataset_name, data_dir):
    print(f"\n📂 正在加载数据集: [{dataset_name.upper()}] ...")
    if dataset_name == 'bloodmnist':
        if medmnist is None:
            raise ImportError("请先安装 medmnist 库！")
        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = DataClass(split='test', transform=transform, download=False, size=224, root=data_dir)
        class_names = list(info['label'].values())
        labels = np.array([label[0] for _, label in dataset]) 
    elif dataset_name in ['neu_cls', 'cub_200', 'dermnet']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        class_names = [clean_classname(cls, dataset_name) for cls in dataset.classes]
        labels = np.array(dataset.targets)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")
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

    # 1. 统一加载与切分
    dataset, class_names, all_labels = load_dataset(d_name, data_path)
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
        lora_alpha=2,                # 通常 alpha=r
        target_modules=["q_proj", "k_proj", "v_proj"], # 注入到 Query, Key, Value 
        lora_dropout=0.25,           # 论文指定 dropout 为 0.25 
        bias="none"
    )
    # peft 默认会匹配所有包含 q/k/v_proj 的层，即 Vision 和 Text 编码器都会被注入 
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)
    model.print_trainable_parameters()

    # 4. 设置优化器与学习率调度器
    # 论文指定学习率 2e-4，并使用余弦退火调度器 
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_subset = Subset(dataset, train_indices)
    train_bs = min(BATCH_SIZE, len(train_indices))
    train_loader = DataLoader(train_subset, batch_size=train_bs, shuffle=True)

    # 论文指定总迭代次数为：500 * (每类样本数) 
    total_iterations = 500 * k_shots
    epochs = total_iterations // len(train_loader) + 1
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n🔥 开始严格复现版 {k_shots}-shot 训练...")
    print(f"   总迭代次数: {total_iterations}, 折合 Epochs: {epochs}")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.view(-1).long().to(DEVICE)
            
            optimizer.zero_grad()
            image_inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(DEVICE)
            
            # 🔥 必须在循环内同时计算图文特征，因为双端 LoRA 都在更新 [cite: 192, 193]
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 还原原生 CLIP 的相似度计算方式 (使用模型自带的可学习温度系数) 
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
            
        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
            print(f"   Epoch {epoch+1:04d}/{epochs} | Loss: {total_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 5. 测试评估
    print("\n📊 正在测试集上评估...")
    model.eval()
    
    # 评估前提取一次最新的文本特征即可
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            image_inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(DEVICE)
            
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.view(-1).numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"\n>>> [{d_name.upper()}] {k_shots}-Shot CLIP-LORA 复现版准确率: {acc * 100:.2f}% <<<")