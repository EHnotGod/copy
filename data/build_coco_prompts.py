import json, random, os
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

random.seed(0)
np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "annotations/captions_val2014.json")
OUT = os.path.join(HERE, "coco_prompts_2000.json")

TARGET = 2000
SIM_THRESH = 0.75  # 论文: pairwise CLIP similarity < 0.75

print("加载 caption...")
data = json.load(open(CAP))
# 每张图只取第一条 caption，先去掉近重复，减小候选池
by_img = {}
for a in data["annotations"]:
    iid = a["image_id"]
    if iid not in by_img:
        by_img[iid] = a["caption"].strip().replace("\n", " ")
caps = list(by_img.values())
# 精确去重
seen = set(); uniq = []
for c in caps:
    k = c.lower()
    if k not in seen:
        seen.add(k); uniq.append(c)
random.shuffle(uniq)
print(f"候选 caption 数（每图一条、精确去重后）: {len(uniq)}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}，加载 CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def encode(texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        b = texts[i:i+bs]
        inp = tok(b, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
        f = model.get_text_features(**inp)
        if not torch.is_tensor(f):
            f = f.pooler_output if hasattr(f, "pooler_output") else f[1]
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu().numpy())
    return np.concatenate(out, 0)

print("编码候选池...")
emb = encode(uniq)  # (N, D) 已归一化

# 贪心去重：逐条加入，与已保留集合最大余弦相似度 < 0.75 才保留
print("贪心去重 (pairwise CLIP < 0.75)...")
kept_idx = []
kept_emb = np.zeros((TARGET, emb.shape[1]), dtype=np.float32)
nk = 0
for i in range(len(uniq)):
    if nk == 0:
        kept_emb[0] = emb[i]; kept_idx.append(i); nk = 1; continue
    sims = kept_emb[:nk] @ emb[i]
    if sims.max() < SIM_THRESH:
        kept_emb[nk] = emb[i]; kept_idx.append(i); nk += 1
        if nk >= TARGET:
            break
print(f"保留 {nk} 条（阈值 {SIM_THRESH}，扫描 {i+1} 条候选）")

prompts = [uniq[j] for j in kept_idx]
records = [{"case_id": str(k), "prompt": p} for k, p in enumerate(prompts)]
json.dump(records, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"写出 {len(records)} 条 -> {OUT}")
print("样例:")
for r in records[:5]:
    print(" ", r["case_id"], r["prompt"])
