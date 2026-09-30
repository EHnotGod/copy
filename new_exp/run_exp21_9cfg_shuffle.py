#!/usr/bin/env python3
"""
§21 9-cfg 跨模型路由 Shuffle 消融（对应 exp7 的 9-cfg 版本）

验证预测器依赖 prompt 级排序信号：
  - Real          : 正常训练
  - Intra-shuffle : 类别内打乱质量标签
  - Full-shuffle  : 全局打乱

类别（gm/am）: position / count / other（关键词匹配）
UniGen: 只跑 real vs full_shuffle

固定: n=2200, score mode only (9-cfg), 3 seeds, budget=10.0
"""
import os, sys, json, random, argparse
import numpy as np
import experiment_integrity as integrity
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sklearn.neighbors
from collections import defaultdict

HERE  = os.path.dirname(os.path.abspath(__file__))
RES = integrity.results_dir(); os.makedirs(RES, exist_ok=True)
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
from embed_lr_dp import load_cached_embedding
import catimage_tf_mlp as TF

DATA      = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
BBASE     = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results/geneval2_boogu/eval"
UBASE     = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results"
CACHE     = os.path.join(CODES, ".cache_embed")
EMB_MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"
CLIP_DIR  = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"

MODELS   = ["qwen", "flux", "boogu"]
CFGS3    = ["10steps", "50steps", "tts"]
DATASETS = ["gm", "am", "unigen"]
SEEDS    = [0, 1, 2]
N_EXT_VAL = 200
N_TRAIN   = 2200
BUDGET    = 10.0
COSTS_9   = np.array([2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8])
QWEN50_IDX = 1

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))

POS_KW = ['above', 'below', 'on the left', 'on the right', 'to the left', 'to the right',
          'ahead of', 'behind', 'leaping over', 'beside', 'in front of', 'next to',
          'left side', 'right side']
NUM_KW = ['two ', 'three ', 'four ', 'five ', 'six ', 'seven ', 'eight ', 'nine ', 'ten ']

def assign_category(instr):
    t = instr.lower()
    if any(k in t for k in POS_KW): return 'position'
    if any(k in t for k in NUM_KW): return 'count'
    return 'other'


def train_paths(dataset, model):
    if dataset == "gm":
        if model == "boogu": return f"{BBASE}/gen2400_perseed.json", None
        return (f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen_perseed.json",
                f"{DATA}/geneval2/edit/geneval2_{model}_edit_gen2000_perseed.json")
    if dataset == "am":
        if model == "boogu": return f"{BBASE}/gen2400_am_v2_perseed.json", None
        return (f"{DATA}/geneval2/edit_am/geneval2_{model}_edit_gen_perseed.json",
                f"{DATA}/geneval2/edit_am/geneval2_{model}_edit_gen2000_perseed.json")
    if model == "boogu": return f"{UBASE}/unigenbench_boogu/eval/gen2400_tts_edit_v2.json", None
    return f"{UBASE}/unigen_{model}/eval/gen2400_edit_v2.json", None

def test_path(dataset, model):
    if dataset == "gm":
        if model == "boogu": return f"{BBASE}/raw800_perseed.json"
        return f"{DATA}/geneval2/edit/geneval2_{model}_edit_raw_perseed.json"
    if dataset == "am":
        if model == "boogu": return f"{BBASE}/raw800_am_perseed.json"
        return f"{DATA}/geneval2/edit_am/geneval2_{model}_edit_raw_perseed.json"
    if model == "boogu": return f"{UBASE}/unigenbench_boogu/eval/raw600_tts_edit_v2.json"
    return f"{UBASE}/unigen_{model}/eval/raw600_edit_v2.json"

def load_scores_single(path):
    d = json.load(open(path))
    out = {}
    for x in d:
        sc = x["scores"]; vals = {}; ok = True
        for c in CFGS3:
            v = sc.get(c)
            if v is None: ok = False; break
            if isinstance(v, dict): v = v.get("avg")
            if v is None: ok = False; break
            vals[c] = float(v)
        if ok: out[x["instruction"]] = vals
    return out

def merge_model_data(dataset, model):
    p1, p2 = train_paths(dataset, model)
    d1 = load_scores_single(p1)
    if p2 and os.path.exists(p2): d1.update(load_scores_single(p2))
    return d1

def load_9cfg_pool(dataset):
    data = {}
    for model in MODELS:
        md = merge_model_data(dataset, model)
        for instr, vals in md.items():
            if instr not in data: data[instr] = {}
            data[instr][model] = vals
    instrs = sorted(instr for instr, v in data.items() if len(v) == 3)
    scores9 = np.zeros((len(instrs), 9), dtype=np.float32)
    for i, instr in enumerate(instrs):
        col = 0
        for model in MODELS:
            for cfg in CFGS3:
                scores9[i, col] = data[instr][model][cfg]
                col += 1
    test_instrs, _ = load_9cfg_test(dataset)
    instrs, scores9, _ = integrity.exclude_test_prompts(instrs, scores9, test_instrs)
    return instrs, scores9

def load_9cfg_test(dataset):
    data = {}
    for model in MODELS:
        td = load_scores_single(test_path(dataset, model))
        for instr, vals in td.items():
            if instr not in data: data[instr] = {}
            data[instr][model] = vals
    instrs = sorted(instr for instr, v in data.items() if len(v) == 3)
    scores9 = np.zeros((len(instrs), 9), dtype=np.float32)
    for i, instr in enumerate(instrs):
        col = 0
        for model in MODELS:
            for cfg in CFGS3:
                scores9[i, col] = data[instr][model][cfg]
                col += 1
    return instrs, scores9

def get_emb(path):
    e = load_cached_embedding(CACHE, path, EMB_MODEL)
    if e is None: raise RuntimeError(f"embed cache miss: {path}")
    return e

def get_pool_emb(dataset):
    instructions, _ = load_9cfg_pool(dataset)
    return integrity.align_embeddings(train_paths(dataset, "qwen"), instructions, get_emb)


def mean_pool(feat, mask):
    m = mask[:, :, None]
    return (feat * m).sum(1) / np.clip(m.sum(1), 1e-6, None)

def fit_ridge(Xtr, ytr, Xva, yva, Xte):
    best_sc, best_m = -1e18, None
    for a in [1., 10., 100., 1000.]:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m.predict(Xte)

def route_quality(pred_mat, true_mat):
    from run_exp25_cross_dataset_generalization import route_without_target_labels
    return float(route_without_target_labels(pred_mat, true_mat, COSTS_9, BUDGET).quality)

def intra_shuffle(s9, instrs, rng):
    cats = [assign_category(instr) for instr in instrs]
    s9_shuf = s9.copy()
    for cat in set(cats):
        idxs = [i for i, c in enumerate(cats) if c == cat]
        perm  = idxs[:]
        rng.shuffle(perm)
        s9_shuf[idxs] = s9[perm]
    return s9_shuf

def full_shuffle(s9, rng):
    idx = list(range(len(s9))); rng.shuffle(idx)
    return s9[idx]


def run_dataset(dataset, gpu, skip_tf=False):
    print(f"\n=== 9-cfg shuffle: {dataset}  gpu={gpu} ===", flush=True)
    dev = f"cuda:{gpu}"
    has_categories = dataset in ("gm", "am")

    pool_instrs, pool_s9 = load_9cfg_pool(dataset)
    te_instrs,   te_s9   = load_9cfg_test(dataset)
    n_pool = len(pool_instrs); n_te = len(te_instrs)
    emb_pool = get_pool_emb(dataset)
    emb_te   = integrity.align_embeddings((test_path(dataset, "qwen"),), te_instrs, get_emb)
    print(f"  pool={n_pool}  test={n_te}", flush=True)

    pfx = f"exp21_{dataset}_"
    Fpool, Mpool = TF.clip_token_features(pool_instrs, pfx+"pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   pfx+"test", CLIP_DIR, dev)
    Cte = mean_pool(Fte, Mte)

    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]
    s9_va   = pool_s9[ext_val]
    Fva, Mva = Fpool[ext_val], Mpool[ext_val]

    true_mat = te_s9
    oracle   = route_quality(true_mat, true_mat)
    always_qwen50 = float(te_s9[:, QWEN50_IDX].mean())
    print(f"  oracle={oracle:.4f}  always_qwen50={always_qwen50:.4f}", flush=True)

    conditions = ["real", "full_shuffle"]
    if has_categories:
        conditions.insert(1, "intra_shuffle")

    cond_rows = {c: {"ridge": [], "knn": []} for c in conditions}

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        sub_instrs = [pool_instrs[i] for i in sub]

        for cond in conditions:
            if cond == "real":
                s9_tr = pool_s9[sub]
            elif cond == "intra_shuffle":
                s9_tr = intra_shuffle(pool_s9[sub], sub_instrs, random.Random(seed+100))
            else:
                s9_tr = full_shuffle(pool_s9[sub], random.Random(seed+200))

            # Ridge
            sc_r  = StandardScaler().fit(emb_pool[sub])
            Xtr_r = sc_r.transform(emb_pool[sub])
            Xva_r = sc_r.transform(emb_pool[ext_val])
            Xte_r = sc_r.transform(emb_te)
            r_preds = np.stack([fit_ridge(Xtr_r, s9_tr[:, j], Xva_r, s9_va[:, j], Xte_r)
                                for j in range(9)], axis=1)
            cond_rows[cond]["ridge"].append(route_quality(r_preds, true_mat))

            # KNN
            kn    = min(50, max(1, len(sub) // 3))
            sc_c  = StandardScaler().fit(mean_pool(Fpool[sub], Mpool[sub]))
            Ctr_s = sc_c.transform(mean_pool(Fpool[sub], Mpool[sub]))
            Cte_s = sc_c.transform(Cte)
            nn_idx = sklearn.neighbors.NearestNeighbors(n_neighbors=kn).fit(Ctr_s).kneighbors(Cte_s, return_distance=False)
            knn_preds = np.stack([s9_tr[nn_idx[:, :], j].mean(1) for j in range(9)], axis=1)
            cond_rows[cond]["knn"].append(route_quality(knn_preds, true_mat))

        print(f"  seed={seed} done", flush=True)

    result = {"dataset": dataset, "oracle": oracle, "always_qwen50": always_qwen50}
    for cond, rows in cond_rows.items():
        result[cond] = {
            "ridge_mean": float(np.mean(rows["ridge"])),
            "ridge_std":  float(np.std(rows["ridge"])),
            "knn_mean":   float(np.mean(rows["knn"])),
            "knn_std":    float(np.std(rows["knn"])),
        }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--skip_tf",  action="store_true")
    ap.add_argument("--out_suffix", default="")
    a = ap.parse_args()

    save_path = os.path.join(RES, f"exp21_9cfg_shuffle{a.out_suffix}.json")
    if os.path.exists(save_path):
        all_results = json.load(open(save_path))
        done = {r["dataset"] for r in all_results}
    else:
        all_results = []; done = set()

    for dataset in a.datasets:
        if dataset in done:
            print(f"[skip] {dataset}"); continue
        result = run_dataset(dataset, a.gpu, a.skip_tf)
        all_results.append(result)
        done.add(dataset)
        json.dump(all_results, open(save_path, "w"), indent=2)
        print(f"[saved] {save_path}", flush=True)
    print(f"[done] {save_path}")
