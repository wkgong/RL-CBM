import os
import torch
import numpy as np
import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 🛑 配置中心
# ==========================================
BLOODMNIST_DIR = "./data"
SHOTS_PER_CLASS = 4          # 👈 每个类别抽取的样本数，可改为 1, 2, 4, 8, 16...
NUM_SEEDS = 3                 # 随机种子数，多次取平均更稳健

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "../RL-CLEAN/openai/clip-vit-large-patch14"

def get_bloodmnist_dataset(npz_folder_path):
    info = INFO['bloodmnist']
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])
    
    try:
        # 加载测试集用于特征提取 (我们在代码里手动切分 1-shot)
        dataset = DataClass(split='test', transform=transform, download=False, size=224, root=npz_folder_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件！请确认 {npz_folder_path} 文件夹下有 bloodmnist.npz 文件。")
        return None, None
    class_names = list(info['label'].values())
    return dataset, class_names

def extract_features(dataset, model, processor):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_features = []
    all_labels = []
    
    print("  正在提取所有图像的 CLIP 视觉特征...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Extracting"):
            image_inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            features = model.get_image_features(**image_inputs)
            # L2 归一化 (关键步骤)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_features), np.array(all_labels)

if __name__ == "__main__":
    print(f"🚀 初始化 {MODEL_NAME} on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    dataset, class_names = get_bloodmnist_dataset(BLOODMNIST_DIR)
    if dataset is None:
        exit()

    # 1. 提取全局特征
    features, labels = extract_features(dataset, model, processor)
    # BloodMNIST 标签可能是 shape (N, 1)，展平它
    if len(labels.shape) > 1:
        labels = labels.ravel()

    print("\n" + "="*60)
    print(f"🧪 开始 {SHOTS_PER_CLASS}-Shot Linear Probing 实验 ({NUM_SEEDS} Seeds)")
    print("="*60)

    all_accs = []
    for seed in range(NUM_SEEDS):
        np.random.seed(42 + seed)

        # 2. 构造 K-shot 训练集 (分层抽样：每个类别严格抽 SHOTS_PER_CLASS 个样本)
        train_idx = []
        test_idx = []
        for c in range(len(class_names)):
            c_idx = np.where(labels == c)[0]
            np.random.shuffle(c_idx)
            num_to_take = min(SHOTS_PER_CLASS, len(c_idx))
            train_idx.extend(c_idx[:num_to_take])
            test_idx.extend(c_idx[num_to_take:])

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_test = features[test_idx]
        y_test = labels[test_idx]

        print(f"\n  🎲 Seed {seed} | 训练集: {len(X_train)} | 测试集: {len(X_test)}")

        # 3. 训练逻辑回归分类器
        clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        # 4. 在测试集上评估
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        all_accs.append(acc)

        print(f"    >>> Seed {seed} 准确率: {acc * 100:.2f}% <<<")

    # 汇总统计
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    print(f"\n{'='*60}")
    print(f"📊 [{SHOTS_PER_CLASS}-Shot Linear Probing] 平均准确率: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"   各Seed结果: {[f'{a*100:.2f}%' for a in all_accs]}")
    print("="*60)