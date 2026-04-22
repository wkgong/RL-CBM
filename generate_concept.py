import json
import os
# 如果你使用的是 OpenAI 接口
from openai import OpenAI

# ==========================================
# 🛑 TODO: 填入你的 API Key 和 Base URL
# ==========================================
client = OpenAI(
    api_key="sk-xxxxxxxxx", # 替换为你的大模型 API Key
    # base_url="https://api.openai.com/v1" # 如果有国内代理地址，填在这里
)
MODEL_NAME = "gpt-4o" # 或者使用 gpt-4-turbo, claude-3-opus 等

# Kaggle 版 DermNet 的 23 个标准大类
DERMNET_CLASSES = [
    "Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections", "Urticaria Hives",
    "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections"
]

def generate_concepts_for_disease(disease_name):
    print(f"🧠 正在请求大模型解析: {disease_name} ...")
    
    prompt = f"""
    You are an expert dermatologist. We are building a visual concept dictionary for a zero-shot vision-language model to identify skin diseases.
    Please break down the skin disease category '{disease_name}' into 3 distinct visual dimensions: 
    1. 'texture' (e.g., scaly, raised, smooth, blistered)
    2. 'color' (e.g., erythematous, silvery, hyperpigmented)
    3. 'shape' (e.g., irregular borders, coin-shaped, clustered)
    
    Return ONLY a valid JSON object strictly in this format, with 3 to 5 short visual phrases per dimension:
    {{
        "texture": ["phrase 1", "phrase 2", "phrase 3"],
        "color": ["phrase 1", "phrase 2", "phrase 3"],
        "shape": ["phrase 1", "phrase 2", "phrase 3"]
    }}
    Do not include any other text or markdown formatting.
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    try:
        # 清理可能带有的 markdown 标记
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        return json.loads(content)
    except Exception as e:
        print(f"❌ 解析 {disease_name} 失败: {e}")
        return {"texture": [], "color": [], "shape": []}

if __name__ == "__main__":
    concept_dict = {}
    
    for cls in DERMNET_CLASSES:
        concepts = generate_concepts_for_disease(cls)
        concept_dict[cls] = concepts
        
    save_path = "dermnet_concepts.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(concept_dict, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 完美！皮肤病概念字典已生成并保存至 {save_path}！")
    print(f"一共包含了 {len(concept_dict)} 个类别的专业视觉特征。")