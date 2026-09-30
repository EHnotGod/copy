#!/usr/bin/env python3
"""
§19 9-cfg 跨模型路由 Seed Sensitivity（对应 exp5 的 9-cfg 版本）

固定 n=2200，对每个 k=1..5 枚举所有 C(5,k) 图片 seed 组合，
每组合 × 3 train/val split seeds 跑 Ridge/KNN 预测器。
test 固定用 5-seed avg 作为 ground truth。
9-cfg 路由：qwen/flux/boogu × 10/50/tts，budget=10.0。

数据集：gm, am（unigen 无 per-seed 差异，暂不跑）
"""
import os, sys, json, random, argparse
from itertools import combinations
import numpy as np
import experiment_integrity as integrity
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sklearn.neighbors

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

MODELS      = ["qwen", "flux", "boogu"]
CFGS3       = ["10steps", "50steps", "tts"]
DATASETS    = ["gm", "am"]
N_FIXED     = 2200
N_EXT_VAL   = 200
TRAIN_SEEDS = list(range(10))
N_IMG_SEEDS = 5
K_LIST      = [1, 2, 3, 4, 5]
BUDGET      = 10.0
COSTS_9     = np.array([2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8])
QWEN50_IDX  = 1

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))


# ── 数据路径 ─────────────────────────────────────────────────
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


def load_perseed_single(path):
    """返回 {instr: {cfg: [s0..s4]}}"""
    d = json.load(open(path))
    out = {}
    for x in d:
        sc = x["scores"]; ok = True; vals = {}
        for c in CFGS3:
            v = sc.get(c)
            if v is None or not isinstance(v, dict): ok = False; break
            ps = v.get("per_seed", {})
            sv = [ps[k] for k in sorted(ps.keys())]
            if len(sv) < N_IMG_SEEDS: ok = False; break
            vals[c] = sv[:N_IMG_SEEDS]
        if ok:
            out[x["instruction"]] = vals
    return out


def merge_perseed_model(dataset, model):
    p1, p2 = train_paths(dataset, model)
    d1 = load_perseed_single(p1)
    if p2 and os.path.exists(p2):
        d1.update(load_perseed_single(p2))
    return d1


def load_9cfg_pool_perseed(dataset):
    """returns (instrs, scores9) where scores9.shape = (N, 9, 5)"""
    data = {}
    for model in MODELS:
        md = merge_perseed_model(dataset, model)
        for instr, vals in md.items():
            if instr not in data: data[instr] = {}
            data[instr][model] = vals
    instrs = sorted(instr for instr, v in data.items() if len(v) == 3)
    scores9 = np.zeros((len(instrs), 9, N_IMG_SEEDS), dtype=np.float32)
    for i, instr in enumerate(instrs):
        col = 0
        for model in MODELS:
            for cfg in CFGS3:
                scores9[i, col, :] = data[instr][model][cfg]
                col += 1
    test_instrs, _ = load_9cfg_test_perseed(dataset)
    instrs, scores9, _ = integrity.exclude_test_prompts(instrs, scores9, test_instrs)
    return instrs, scores9  # (N, 9, 5)


def load_9cfg_test_perseed(dataset):
    data = {}
    for model in MODELS:
        td = load_perseed_single(test_path(dataset, model))
        for instr, vals in td.items():
            if instr not in data: data[instr] = {}
            data[instr][model] = vals
    instrs = sorted(instr for instr, v in data.items() if len(v) == 3)
    scores9 = np.zeros((len(instrs), 9, N_IMG_SEEDS), dtype=np.float32)
    for i, instr in enumerate(instrs):
        col = 0
        for model in MODELS:
            for cfg in CFGS3:
                scores9[i, col, :] = data[instr][model][cfg]
                col += 1
    return instrs, scores9


def get_emb(path):
    e = load_cached_embedding(CACHE, path, EMB_MODEL)
    if e is None: raise RuntimeError(f"embed cache miss: {path}")
    return e

def get_pool_emb(dataset):
    instructions, _ = load_9cfg_pool_perseed(dataset)
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
    print(f"\n=== 9-cfg seed-sens: {dataset}  gpu={gpu} ===", flush=True)
    dev = f"cuda:{gpu}"

    pool_instrs, pool_ps9 = load_9cfg_pool_perseed(dataset)  # (N, 9, 5)
    te_instrs,   te_ps9   = load_9cfg_test_perseed(dataset)
    n_pool = len(pool_instrs); n_te = len(te_instrs)
    print(f"  pool={n_pool}  test={n_te}", flush=True)

    emb_pool = get_pool_emb(dataset)
    emb_te   = integrity.align_embeddings((test_path(dataset, "qwen"),), te_instrs, get_emb)

    pfx = f"exp19_{dataset}_"
    Fpool, Mpool = TF.clip_token_features(pool_instrs, pfx+"pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   pfx+"test", CLIP_DIR, dev)
    Cte = mean_pool(Fte, Mte)

    # true_mat from 5-seed avg
    true_mat = te_ps9.mean(2)  # (N_te, 9)

    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]

    oracle = route_quality(true_mat, true_mat)
    always_qwen50 = float(true_mat[:, QWEN50_IDX].mean())
    print(f"  oracle={oracle:.4f}  always_qwen50={always_qwen50:.4f}", flush=True)

    k_data = []
    for k in K_LIST:
        print(f"  k={k}", end="  ", flush=True)
        # val scores for this k
        s9_va_k = pool_ps9[ext_val, :, :k].mean(2)  # (n_val, 9)

        seed_rows = {"ridge_score": [], "knn_score": []}

        for seed in TRAIN_SEEDS:
            rng = random.Random(seed)
            sub = np.array(rng.sample(lc_pool, min(N_FIXED, len(lc_pool))))
            s9_tr_k = pool_ps9[sub, :, :k].mean(2)  # (n, 9)

            # Ridge score
            sc_r  = StandardScaler().fit(emb_pool[sub])
            Xtr_r = sc_r.transform(emb_pool[sub])
            Xva_r = sc_r.transform(emb_pool[ext_val])
            Xte_r = sc_r.transform(emb_te)
            r_preds = np.stack([fit_ridge(Xtr_r, s9_tr_k[:, j], Xva_r, s9_va_k[:, j], Xte_r)
                                for j in range(9)], axis=1)
            seed_rows["ridge_score"].append(route_quality(r_preds, true_mat))

            # KNN score
            kn    = min(50, max(1, len(sub) // 3))
            sc_c  = StandardScaler().fit(mean_pool(Fpool[sub], Mpool[sub]))
            Ctr_s = sc_c.transform(mean_pool(Fpool[sub], Mpool[sub]))
            Cte_s = sc_c.transform(Cte)
            nn_idx = sklearn.neighbors.NearestNeighbors(n_neighbors=kn).fit(Ctr_s).kneighbors(Cte_s, return_distance=False)
            knn_preds = np.stack([s9_tr_k[nn_idx[:, :], j].mean(1) for j in range(9)], axis=1)
            seed_rows["knn_score"].append(route_quality(knn_preds, true_mat))

        print("done", flush=True)
        pt = {"k": k}
        for key, vals in seed_rows.items():
            pt[f"{key}_mean"] = float(np.mean(vals))
            pt[f"{key}_std"]  = float(np.std(vals))
        k_data.append(pt)

    return {"dataset": dataset, "oracle": oracle,
            "always_qwen50": always_qwen50, "k_data": k_data}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--skip_tf",  action="store_true")
    ap.add_argument("--out_suffix", default="")
    a = ap.parse_args()

    save_path = os.path.join(RES, f"exp19_9cfg_seed_sens{a.out_suffix}.json")
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
