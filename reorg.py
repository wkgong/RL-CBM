import os
import shutil
from tqdm import tqdm

# ==========================================
# 🛑 TODO: 修改为你当前所有图片混在一起的文件夹路径
# ==========================================
SOURCE_DIR = "./data/NEU-CLS" 

# ==========================================
# 🛑 TODO: 重组后生成的新文件夹路径 (给 ImageFolder 用的)
# ==========================================
TARGET_DIR = "./data/NEU-CLS-ImageFolder"

# 前缀到标准类别名的映射字典
CLASS_MAP = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches"
}

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 找不到源文件夹: {SOURCE_DIR}")
        exit()

    # 1. 在目标路径下创建 6 个子文件夹
    for class_name in CLASS_MAP.values():
        os.makedirs(os.path.join(TARGET_DIR, class_name), exist_ok=True)

    # 2. 遍历并复制文件
    files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    print(f"📂 在源文件夹中找到 {len(files)} 张图片，开始分类重组...")

    success_count = 0
    for filename in tqdm(files):
        # 提取下划线前的前缀，比如从 "Sc_203.bmp" 提取 "Sc"
        prefix = filename.split('_')[0]
        
        if prefix in CLASS_MAP:
            target_class_folder = CLASS_MAP[prefix]
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(TARGET_DIR, target_class_folder, filename)
            
            # 使用 copy2 保留元数据 (如果你想直接移动以节省空间，可以换成 shutil.move)
            shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            print(f"\n⚠️ 警告: 无法识别文件 {filename} 的类别前缀 '{prefix}'，跳过。")

    print(f"\n✅ 重组完成！成功处理 {success_count} 张图片。")
    print(f"👉 请将你在 Master 脚本中的 CONFIG['paths']['neu_cls'] 修改为: {TARGET_DIR}")