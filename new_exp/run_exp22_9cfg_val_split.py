#!/usr/bin/env python3
"""
§22 9-cfg 跨模型路由 Val-split（对应 exp9 的 9-cfg 版本）

固定总池 2400，改变 val 大小：200/400/800。
score mode only，9-cfg Lagrangian routing，budget=10.0。
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

MODELS    = ["qwen", "flux", "boogu"]
CFGS3     = ["10steps", "50steps", "tts"]
DATASETS  = ["gm", "am", "unigen"]
SEEDS     = [0, 1, 2]
VAL_SIZES = [200, 400, 800]
BUDGET    = 10.0
COSTS_9   = np.array([2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8])
QWEN50_IDX = 1

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))


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


def run_dataset(dataset, gpu, skip_tf=False):
    print(f"\n=== 9-cfg val-split: {dataset}  gpu={gpu} ===", flush=True)
    dev = f"cuda:{gpu}"

    pool_instrs, pool_s9 = load_9cfg_pool(dataset)
    te_instrs,   te_s9   = load_9cfg_test(dataset)
    n_pool = len(pool_instrs); n_te = len(te_instrs)
    emb_pool = get_pool_emb(dataset)
    emb_te   = integrity.align_embeddings((test_path(dataset, "qwen"),), te_instrs, get_emb)
    print(f"  pool={n_pool}  test={n_te}", flush=True)

    pfx = f"exp22_{dataset}_"
    Fpool, Mpool = TF.clip_token_features(pool_instrs, pfx+"pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   pfx+"test", CLIP_DIR, dev)
    Cte = mean_pool(Fte, Mte)

    true_mat = te_s9
    oracle   = route_quality(true_mat, true_mat)
    always_qwen50 = float(te_s9[:, QWEN50_IDX].mean())
    print(f"  oracle={oracle:.4f}  always_qwen50={always_qwen50:.4f}", flush=True)

    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)

    split_data = []
    for val_size in VAL_SIZES:
        n_train = len(all_idx) - val_size
        print(f"  val={val_size}  train={n_train}", end="  ", flush=True)
        ridge_vals = []; knn_vals = []; tf_vals = []

        for seed in SEEDS:
            rng = random.Random(seed + 1000)
            idx_perm = all_idx[:]
            rng.shuffle(idx_perm)
            ext_val = np.array(idx_perm[:val_size])
            sub     = np.array(idx_perm[val_size:val_size + n_train])
            s9_va   = pool_s9[ext_val]
            s9_tr   = pool_s9[sub]
            Fva, Mva = Fpool[ext_val], Mpool[ext_val]

            # Ridge
            sc_r  = StandardScaler().fit(emb_pool[sub])
            Xtr_r = sc_r.transform(emb_pool[sub])
            Xva_r = sc_r.transform(emb_pool[ext_val])
            Xte_r = sc_r.transform(emb_te)
            r_preds = np.stack([fit_ridge(Xtr_r, s9_tr[:, j], Xva_r, s9_va[:, j], Xte_r)
                                for j in range(9)], axis=1)
            ridge_vals.append(route_quality(r_preds, true_mat))

            # KNN
            kn    = min(50, max(1, len(sub) // 3))
            sc_c  = StandardScaler().fit(mean_pool(Fpool[sub], Mpool[sub]))
            Ctr_s = sc_c.transform(mean_pool(Fpool[sub], Mpool[sub]))
            Cte_s = sc_c.transform(Cte)
            nn_idx = sklearn.neighbors.NearestNeighbors(n_neighbors=kn).fit(Ctr_s).kneighbors(Cte_s, return_distance=False)
            knn_preds = np.stack([s9_tr[nn_idx[:, :], j].mean(1) for j in range(9)], axis=1)
            knn_vals.append(route_quality(knn_preds, true_mat))

            # TFMLP
            if not skip_tf:
                lo9 = s9_tr.min(0); hi9 = s9_tr.max(0); rng9 = np.maximum(hi9-lo9, 1e-6)
                s9_tr_n = (s9_tr-lo9)/rng9; s9_va_n = (s9_va-lo9)/rng9
                pred_n, _, _ = TF.train_tf_mlp(
                    Fpool[sub], Mpool[sub], s9_tr_n,
                    Fva, Mva, s9_va_n,
                    Fte, Mte, M=9, seed=seed, device=dev)
                tf_vals.append(route_quality(pred_n*rng9+lo9, true_mat))

        print("done", flush=True)
        pt = {"val_size": val_size, "n_train": n_train,
              "ridge_mean": float(np.mean(ridge_vals)), "ridge_std": float(np.std(ridge_vals)),
              "knn_mean":   float(np.mean(knn_vals)),   "knn_std":   float(np.std(knn_vals))}
        if tf_vals:
            pt["tf_mean"] = float(np.mean(tf_vals)); pt["tf_std"] = float(np.std(tf_vals))
        split_data.append(pt)

    return {"dataset": dataset, "oracle": oracle,
            "always_qwen50": always_qwen50, "split_data": split_data}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--skip_tf",  action="store_true")
    ap.add_argument("--out_suffix", default="")
    a = ap.parse_args()

    save_path = os.path.join(RES, f"exp22_9cfg_val_split{a.out_suffix}.json")
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
