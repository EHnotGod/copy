#!/usr/bin/env python3
"""
§18 9-cfg 跨模型路由 LC

在同一个 dataset 内，把 qwen/flux/boogu × 10steps/50steps/tts = 9 个 cfg
放入同一个路由池，对每条 query 选最优 cfg。

Score mode  : 预测 9 个绝对分数，(N,9) pred_mat，Lagrangian routing
Delta mode  : 预测每个 cfg 相对 qwen50 的差值，(N,9) pred_mat
              qwen50 位置对应的预测设为 0（anchor），其余预测差值

Budget = 10.0（= qwen 50steps cost）对所有 dataset/model 统一。

cfgs / costs（9 个）:
  qwen10=2.0  qwen50=10.0  qwentts=30.5
  flux10=4.7  flux50=23.1  fluxtts=44.9
  boogu10=4.3 boogu50=21.5 boogutts=64.8

输出 JSON 每条: {dataset, oracle, lp, always_qwen50, lc=[{n, ridge_score,
  ridge_delta, knn_score, knn_delta, tf_score, tf_delta, ...seeds}, ...]}
"""
import os, sys, json, random, argparse
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

DATASETS = ["gm", "am", "unigen"]
MODELS   = ["qwen", "flux", "boogu"]
CFGS3    = ["10steps", "50steps", "tts"]

# 9-cfg 全局定义（顺序：qwen10/50/tts, flux10/50/tts, boogu10/50/tts）
CFG_NAMES = [f"{m}_{c}" for m in MODELS for c in CFGS3]  # 9个
COSTS_9   = np.array([2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8])
BUDGET    = 10.0   # = qwen50 cost，统一 budget
QWEN50_IDX = 1     # CFG_NAMES[1] = qwen_50steps，anchor for delta mode

LC_SIZES  = [100, 300, 600, 1000, 1500, 2200]
SEEDS     = [0, 1, 2]
N_EXT_VAL = 200

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    [10., 100., 1e6],
]))


# ── 数据路径（per model per dataset）──────────────────────────
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
    """load (instruction→{cfg: score}) from one file."""
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
        if ok:
            out[x["instruction"]] = vals
    return out


def merge_model_data(dataset, model):
    """合并 train pool 的两段 (或一段) → {instr: {cfg: score}}"""
    p1, p2 = train_paths(dataset, model)
    d1 = load_scores_single(p1)
    if p2:
        d2 = load_scores_single(p2)
        d1.update(d2)
    return d1


def load_9cfg_pool(dataset):
    """
    返回 (instrs, scores_9) 其中 scores_9.shape = (N, 9)，
    指令顺序 = 三模型 2400条公共指令的交集（实际上就是 2400 条）。
    """
    data = {}  # instr → {model: {cfg: score}}
    for model in MODELS:
        md = merge_model_data(dataset, model)
        for instr, vals in md.items():
            if instr not in data:
                data[instr] = {}
            data[instr][model] = vals

    # 保留三模型都有数据的指令
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
            if instr not in data:
                data[instr] = {}
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


# ── 嵌入缓存（text embedding 与 model 无关，只用 qwen 的路径作 key）──
def get_emb(path):
    e = load_cached_embedding(CACHE, path, EMB_MODEL)
    if e is None: raise RuntimeError(f"embed cache miss: {path}")
    return e

def get_pool_emb(dataset):
    instructions, _ = load_9cfg_pool(dataset)
    return integrity.align_embeddings(train_paths(dataset, "qwen"), instructions, get_emb)


def get_test_emb(dataset):
    instructions, _ = load_9cfg_test(dataset)
    return integrity.align_embeddings((test_path(dataset, "qwen"),), instructions, get_emb)


# ── CLIP features（同 exp4，用 qwen 路径做 cache key）─────────
def get_pool_clip(dataset, dev):
    pool_instrs, _ = load_9cfg_pool(dataset)
    pfx = f"exp18_{dataset}_pool"
    return TF.clip_token_features(pool_instrs, pfx, CLIP_DIR, dev)

def get_test_clip(dataset, dev):
    te_instrs, _ = load_9cfg_test(dataset)
    pfx = f"exp18_{dataset}_test"
    return TF.clip_token_features(te_instrs, pfx, CLIP_DIR, dev)


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


# ── 路由核心（9 cfg 版）──────────────────────────────────────
def route_quality(pred_mat, true_mat, budget):
    from run_exp25_cross_dataset_generalization import route_without_target_labels
    return float(route_without_target_labels(pred_mat, true_mat, COSTS_9, budget).quality)


def lp_quality(means9, budget):
    """全 pool 9个cfg均值的上凸包在 budget 处插值。"""
    from itertools import combinations
    pts = list(zip(COSTS_9.tolist(), means9.tolist()))
    best = -np.inf
    n = len(pts)
    for i in range(n):
        ci, qi = pts[i]
        if abs(ci - budget) < 1e-9:
            best = max(best, qi)
    for i, j in combinations(range(n), 2):
        ci, qi = pts[i]; cj, qj = pts[j]
        lo, hi = (i, j) if ci <= cj else (j, i)
        cl, ql = pts[lo]; ch, qh = pts[hi]
        if cl <= budget <= ch:
            w = (budget - cl) / (ch - cl + 1e-12)
            best = max(best, ql + w * (qh - ql))
    return float(best)


# ── 单 dataset LC ────────────────────────────────────────────
def run_dataset_lc(dataset, gpu, skip_tf=False):
    print(f"\n=== 9-cfg Routing LC: {dataset}  gpu={gpu} ===", flush=True)
    dev = f"cuda:{gpu}"

    # 数据
    pool_instrs, pool_s9 = load_9cfg_pool(dataset)
    te_instrs,   te_s9   = load_9cfg_test(dataset)
    n_pool = len(pool_instrs); n_te = len(te_instrs)
    print(f"  pool={n_pool}  test={n_te}", flush=True)

    # text embeddings (2400 pool)
    emb_pool = get_pool_emb(dataset)
    emb_te   = get_test_emb(dataset)

    # CLIP
    Fpool, Mpool = get_pool_clip(dataset, dev)
    Fte,   Mte   = get_test_clip(dataset, dev)
    Cte = mean_pool(Fte, Mte)

    # fixed val split
    rng_fix = random.Random(42)
    all_idx = list(range(n_pool)); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_EXT_VAL])
    lc_pool = all_idx[N_EXT_VAL:]
    s9_va   = pool_s9[ext_val]
    Fva, Mva = Fpool[ext_val], Mpool[ext_val]

    # Oracle & LP
    oracle = route_quality(te_s9, te_s9, BUDGET)
    from run_exp25_cross_dataset_generalization import fixed_lp
    lp = fixed_lp(pool_s9[np.asarray(lc_pool)], te_s9, COSTS_9, BUDGET).quality
    always_qwen50 = float(te_s9[:, QWEN50_IDX].mean())
    print(f"  oracle={oracle:.4f}  lp={lp:.4f}  always_qwen50={always_qwen50:.4f}", flush=True)

    lc_data = []

    for n in integrity.available_sizes(LC_SIZES, len(lc_pool)):
        print(f"  n={n:5d}", end="  ", flush=True)
        rows = {k: [] for k in ["ridge_score", "ridge_delta",
                                  "knn_score",   "knn_delta",
                                  "tf_score",    "tf_delta"]}

        for seed in SEEDS:
            rng = random.Random(seed)
            sub = np.array(rng.sample(lc_pool, min(n, len(lc_pool))))
            s9_tr = pool_s9[sub]   # (n, 9)

            # ── Ridge ──────────────────────────────────────
            sc_r  = StandardScaler().fit(emb_pool[sub])
            Xtr_r = sc_r.transform(emb_pool[sub])
            Xva_r = sc_r.transform(emb_pool[ext_val])
            Xte_r = sc_r.transform(emb_te)

            # score mode: predict 9 cfgs
            r_preds = np.stack(
                [fit_ridge(Xtr_r, s9_tr[:, j], Xva_r, s9_va[:, j], Xte_r)
                 for j in range(9)], axis=1)  # (N_te, 9)
            rows["ridge_score"].append(route_quality(r_preds, te_s9, BUDGET))

            # delta mode: predict relative to qwen50 (index 1)
            # for each cfg j: delta_j = score_j - score_qwen50
            delta_tr = s9_tr - s9_tr[:, QWEN50_IDX:QWEN50_IDX+1]  # (n, 9), qwen50 col = 0
            delta_va = s9_va - s9_va[:, QWEN50_IDX:QWEN50_IDX+1]
            r_delta_preds = np.stack(
                [fit_ridge(Xtr_r, delta_tr[:, j], Xva_r, delta_va[:, j], Xte_r)
                 for j in range(9)], axis=1)
            rows["ridge_delta"].append(route_quality(r_delta_preds, te_s9, BUDGET))

            # ── KNN ────────────────────────────────────────
            k     = min(50, max(1, n // 3))
            sc_c  = StandardScaler().fit(mean_pool(Fpool[sub], Mpool[sub]))
            Ctr_s = sc_c.transform(mean_pool(Fpool[sub], Mpool[sub]))
            Cte_s = sc_c.transform(Cte)

            nn_idx = sklearn.neighbors.NearestNeighbors(n_neighbors=k).fit(Ctr_s).kneighbors(Cte_s, return_distance=False)

            # knn score mode
            knn_s_preds = np.stack(
                [s9_tr[nn_idx[:, :], j].mean(1) for j in range(9)], axis=1)
            rows["knn_score"].append(route_quality(knn_s_preds, te_s9, BUDGET))

            # knn delta mode
            delta_tr_knn = s9_tr - s9_tr[:, QWEN50_IDX:QWEN50_IDX+1]
            knn_d_preds = np.stack(
                [delta_tr_knn[nn_idx[:, :], j].mean(1) for j in range(9)], axis=1)
            rows["knn_delta"].append(route_quality(knn_d_preds, te_s9, BUDGET))

            # ── TFMLP ──────────────────────────────────────
            if not skip_tf:
                # score mode: predict all 9 at once (M=9)
                s9_tr_n = s9_tr.copy(); s9_va_n = s9_va.copy()
                lo9 = s9_tr_n.min(0); hi9 = s9_tr_n.max(0)
                rng9 = np.maximum(hi9 - lo9, 1e-6)
                s9_tr_n = (s9_tr_n - lo9) / rng9
                s9_va_n = (s9_va_n - lo9) / rng9

                tf_pred_n, _, _ = TF.train_tf_mlp(
                    Fpool[sub], Mpool[sub], s9_tr_n,
                    Fva, Mva, s9_va_n,
                    Fte, Mte, M=9, seed=seed, device=dev)
                tf_score_preds = tf_pred_n * rng9 + lo9
                rows["tf_score"].append(route_quality(tf_score_preds, te_s9, BUDGET))

                # delta mode
                dtr = s9_tr - s9_tr[:, QWEN50_IDX:QWEN50_IDX+1]
                dva = s9_va - s9_va[:, QWEN50_IDX:QWEN50_IDX+1]
                lo_d = dtr.min(0); hi_d = dtr.max(0)
                rd   = np.maximum(hi_d - lo_d, 1e-6)
                dtr_n = (dtr - lo_d) / rd
                dva_n = (dva - lo_d) / rd

                tf_dpred_n, _, _ = TF.train_tf_mlp(
                    Fpool[sub], Mpool[sub], dtr_n,
                    Fva, Mva, dva_n,
                    Fte, Mte, M=9, seed=seed, device=dev)
                tf_delta_preds = tf_dpred_n * rd + lo_d
                rows["tf_delta"].append(route_quality(tf_delta_preds, te_s9, BUDGET))

        print("ridge✓  knn✓" + ("  tf✓" if not skip_tf else "  tf-skip"), flush=True)
        pt = {"n": n}
        for key, vals in rows.items():
            pt[key] = float(np.mean(vals)) if vals else None
            pt[key + "_seeds"] = [float(v) for v in vals]
        lc_data.append(pt)

    return {
        "dataset": dataset,
        "oracle": oracle,
        "lp": lp,
        "lp_protocol": "full construction pool excluding validation; selected without test labels",
        "always_qwen50": always_qwen50,
        "lc": lc_data,
    }


# ── 主入口 ───────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--skip_tf",  action="store_true")
    ap.add_argument("--out_suffix", default="")
    a = ap.parse_args()

    SAVE_PATH = os.path.join(RES, f"exp18_9cfg_routing{a.out_suffix}.json")

    if os.path.exists(SAVE_PATH):
        all_results = json.load(open(SAVE_PATH))
        done = {r["dataset"] for r in all_results}
        print(f"[resume] done={done}")
    else:
        all_results = []; done = set()

    for dataset in a.datasets:
        if dataset in done:
            print(f"[skip] {dataset}"); continue
        result = run_dataset_lc(dataset, a.gpu, a.skip_tf)
        all_results.append(result)
        done.add(dataset)
        json.dump(all_results, open(SAVE_PATH, "w"), indent=2)
        print(f"[saved] {SAVE_PATH}", flush=True)

    print(f"[done] {SAVE_PATH}")
