import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    medmnist = None

# ==========================================
# 🛑 统一配置中心 (Master Config)
# ==========================================
CONFIG = {
    # 1. 选择要跑的数据集: 'bloodmnist', 'neu_cls', 或 'cub_200',dermnet
    "dataset_name": "dermnet",

    # 2. 填写你本地的数据集路径
    "paths": {
        "bloodmnist": "./data",                              # .npz 所在文件夹
        "neu_cls": "./data/NEU-CLS-ImageFolder",             # 缺陷子文件夹的根目录
        "cub_200": "./data/CUB_200_2011/images",              # 鸟类子文件夹的根目录
        "dermnet": "./data/dermnet/test",
    }
}

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14"

# ==========================================
# 📚 统一提示词字典 (Prompt Bank - Ensemble 版)
# ==========================================
PROMPT_BANK = {
    "bloodmnist": [
        "A microscopic image showing a {}.",
        "A micrograph of a {}.",
        "A blood smear image containing a {}.",
        "A zoomed-in pathology photo of a {}.",
        "A medical image of a {}.",
        "A pathology slide showing a {}.",
        "A Wright-Giemsa stained micrograph of a {}.",
        "A close-up view of a {} under a microscope."
    ],
    "neu_cls": [
        "An industrial grayscale photo showing {} defect.",
        "A surface defect of {}.",
        "A zoomed-in image of {} on a steel surface.",
        "A manufacturing defect classification of {}.",
        "A hot-rolled steel strip with {}.",
        "An industrial inspection camera photo of {}."
    ],
    "cub_200": [
        "A wildlife photo of a {}.",
        "A photo of the bird {}.",
        "A nature photograph showing a {}.",
        "A close-up of a {} bird.",
        "An ornithology photo of a {}."
    ],
    "dermnet": [
        "A clinical photograph showing {}.",
        "A dermatology image of a patient with {}.",
        "A close-up skin lesion diagnosed as {}.",
        "A medical photo of {} on the skin."
    ]
}


# ==========================================
# 🛠️ 统一数据加载与清洗模块
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

    elif dataset_name in ['neu_cls', 'cub_200', 'dermnet']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        class_names = [clean_classname(cls, dataset_name) for cls in dataset.classes]

    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, class_names

# ==========================================
# 🔬 核心 Zero-Shot 验证逻辑 (Ensemble 平均池化版)
# ==========================================
def evaluate_zeroshot_clip_ensemble(loader, class_names, templates, model, processor):
    model.eval()
    print(f"  [Ensemble] 正在使用 {len(templates)} 个模板融合文本特征...")

    # 1. 提取多模板文本特征并求平均
    zeroshot_weights = []
    with torch.no_grad():
        for cls in class_names:
            texts = [template.format(cls) for template in templates]
            text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)

            class_embeddings = model.get_text_features(**text_inputs)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm(dim=-1)
            zeroshot_weights.append(class_embedding)

    text_features = torch.stack(zeroshot_weights, dim=0).to(DEVICE)

    all_preds, all_labels = [], []

    # 2. 图像推理
    print("  开始图像推理...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference", leave=False):
            image_inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            preds = logits_per_image.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 3. 输出结果
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n  >>> [{CONFIG['dataset_name'].upper()}] Ensemble Zero-Shot 准确率: {acc * 100:.2f}% <<<")
    print("  详细分类报告 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# ==========================================
# 🚀 主程序
# ==========================================
if __name__ == "__main__":
    d_name = CONFIG["dataset_name"]
    data_path = CONFIG["paths"][d_name]
    templates = PROMPT_BANK[d_name]

    print(f"🚀 初始化 {MODEL_NAME} on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 1. 统一加载数据集
    loader, class_names = load_dataset(d_name, data_path)
    print(f"   识别到 {len(class_names)} 个类别: {class_names[:3]} ...")

    # 2. Zero-Shot 评估
    print(f"\n{'='*60}")
    print(f"🧪 Zero-Shot 测试 [{d_name.upper()}] (Prompt Ensemble 版)")
    print(f"{'='*60}")
    evaluate_zeroshot_clip_ensemble(loader, class_names, templates, model, processor)
