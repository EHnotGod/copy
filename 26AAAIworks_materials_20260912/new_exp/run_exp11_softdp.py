#!/usr/bin/env python3
"""
§11 Soft-DP Loss vs MAE：以背包路由质量为优化目标的 TFMLP

对比：
  tf_mae_score   : 原始 TFMLP（BCEWithLogits，归一化分值），score mode routing
  tf_mae_delta   : 原始 TFMLP，delta mode routing
  tf_dp_score    : Soft-DP TFMLP（loss=soft_dp_loss_exact），预测 3 个绝对分值
  tf_dp_delta    : Soft-DP TFMLP，预测 2 个 delta（d1=q50-q10, d2=qtts-q50）

Ridge（MAE）保持作为对照基线。

固定 n=2200 train, 200 val（seed=42），3 split seeds 均值。
Lagrangian routing 评估路由质量。

输出 JSON: {dataset, model, oracle, lp, always_50steps,
            results=[{predictor, mode, mean, std}]}
"""
import os, sys, json, random, argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

HERE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
from embed_lr_dp import load_cached_embedding
import catimage_tf_mlp as TF

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
BBASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results/geneval2_boogu/eval"
UBASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results"
CACHE = os.path.join(CODES, ".cache_embed")
EMB_MODEL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"
CLIP_DIR  = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"

MODELS   = ["qwen", "flux", "boogu"]
CFGS     = ["10steps", "50steps", "tts"]
DATASETS = ["gm", "am", "unigen"]
SEEDS    = [0, 1, 2]
N_TRAIN  = 2200
N_EXT_VAL = 200

COSTS   = {"qwen": np.array([2.0,10.0,30.5]), "flux": np.array([4.7,23.1,44.9]), "boogu": np.array([4.3,21.5,64.8])}
BUDGETS = {"qwen": 10.0, "flux": 23.1, "boogu": 21.5}

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0,0.05,2000), np.linspace(0.05,0.3,500),
    np.linspace(0.3,3.0,200), [10.,100.,1e6]]))


# ── 数据路径（与 exp4 一致）──────────────────────────────────
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

def load_scores(path):
    d = json.load(open(path))
    instrs, scores = [], {c: [] for c in CFGS}
    for x in d:
        sc = x["scores"]; vals = {}; ok = True
        for c in CFGS:
            v = sc.get(c)
            if v is None: ok = False; break
            if isinstance(v, dict): v = v.get("avg")
            if v is None: ok = False; break
            vals[c] = float(v)
        if not ok: continue
        instrs.append(x["instruction"])
        for c in CFGS: scores[c].append(vals[c])
    return instrs, {c: np.array(scores[c], np.float32) for c in CFGS}

def merge_pool(dataset, model):
    p1, p2 = train_paths(dataset, model)
    i1, s1 = load_scores(p1)
    if p2:
        i2, s2 = load_scores(p2)
        return i1+i2, {c: np.concatenate([s1[c],s2[c]]) for c in CFGS}
    return i1, s1

def get_emb(path):
    e = load_cached_embedding(CACHE, path, EMB_MODEL)
    if e is None: raise RuntimeError(f"embed cache miss: {path}")
    return e

def merge_emb(dataset, model):
    p1, p2 = train_paths(dataset, model)
    e1 = get_emb(p1)
    if p2: return np.concatenate([e1, get_emb(p2)], 0)
    return e1

def mean_pool(feat, mask):
    m = mask[:, :, None]
    return (feat * m).sum(1) / np.clip(m.sum(1), 1e-6, None)

def fit_ridge(Xtr, ytr, Xva, yva, Xte):
    best_sc, best_m = -1e18, None
    for a in [1.,10.,100.,1000.]:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc=sc; best_m=m
    return best_m.predict(Xte)


# ── Lagrangian 路由质量（复用 exp4 实现）──────────────────────
def route_quality(pred_mat, true_mat, costs, budget):
    from collections import defaultdict
    pts = []
    for lam in LAMBDAS:
        adj    = pred_mat - lam * costs[None,:]
        chosen = np.argmax(adj, axis=1)
        avg_c  = float(np.mean(costs[chosen]))
        avg_q  = float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))
        pts.append((avg_c, avg_q))
    bucket = defaultdict(float)
    for c, q in pts:
        bucket[round(c,6)] = max(bucket[round(c,6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if budget <= xs[0]: return float(qs[0])
    if budget >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, budget))
    c0,c1 = xs[idx-1],xs[idx]; q0,q1 = qs[idx-1],qs[idx]
    if c1-c0 < 1e-9: return float(max(q0,q1))
    return float(q0 + (budget-c0)/(c1-c0) * (q1-q0))

def lp_quality(test_means, costs, budget):
    pts = sorted(zip(costs.tolist(), test_means.tolist()))
    best = -np.inf
    for i in range(len(pts)):
        for j in range(len(pts)):
            ci,qi = pts[i]; cj,qj = pts[j]
            if i == j:
                if abs(ci-budget)<1e-9: best=max(best,qi)
                continue
            lo,hi = (i,j) if ci<cj else (j,i)
            cl,ql = pts[lo]; ch,qh = pts[hi]
            if cl<=budget<=ch:
                best = max(best, ql+(budget-cl)/(ch-cl+1e-12)*(qh-ql))
    return float(best)


# ── 主实验 ───────────────────────────────────────────────────
def run_group(dataset, model, gpu):
    import torch
    print(f"\n=== §11 Soft-DP: {dataset}/{model}  gpu={gpu} ===", flush=True)
    dev    = f"cuda:{gpu}"
    costs  = COSTS[model]
    budget = BUDGETS[model]
    int_costs = torch.tensor(np.round(costs*10).astype(int), dtype=torch.int32)

    pool_instrs, pool_scores = merge_pool(dataset, model)
    emb_pool = merge_emb(dataset, model)
    te_path  = test_path(dataset, model)
    te_instrs, yte_cfg = load_scores(te_path)
    n_te = len(te_instrs)
    emb_te = get_emb(te_path)[:n_te]
    print(f"  pool={len(pool_instrs)}  test={n_te}", flush=True)

    pfx = f"exp4_{dataset}_{model}_"
    Fpool, Mpool = TF.clip_token_features(pool_instrs, pfx+"pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   pfx+"test", CLIP_DIR, dev)

    rng_fix = random.Random(42)
    all_idx = list(range(len(pool_instrs))); rng_fix.shuffle(all_idx)
    ext_val  = np.array(all_idx[:N_EXT_VAL])
    lc_pool  = all_idx[N_EXT_VAL:]
    y_va     = {c: pool_scores[c][ext_val] for c in CFGS}
    Fva, Mva = Fpool[ext_val], Mpool[ext_val]

    true_mat = np.stack([yte_cfg[c] for c in CFGS], axis=1)
    oracle   = route_quality(true_mat, true_mat, costs, budget)
    lp       = lp_quality(np.array([yte_cfg[c].mean() for c in CFGS]), costs, budget)
    always_50 = float(yte_cfg["50steps"].mean())
    print(f"  oracle={oracle:.4f}  lp={lp:.4f}  50steps={always_50:.4f}", flush=True)

    rows = {k: [] for k in [
        "ridge_score","ridge_delta",
        "tf_mae_score","tf_mae_delta",
        "tf_dp_score","tf_dp_delta"]}

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        y_tr = {c: pool_scores[c][sub] for c in CFGS}
        d_tr = np.stack([y_tr["50steps"]-y_tr["10steps"],
                         y_tr["tts"]-y_tr["50steps"]], 1)

        # ── Ridge baseline ───────────────────────────────
        sc_r  = StandardScaler().fit(emb_pool[sub])
        Xtr_r = sc_r.transform(emb_pool[sub])
        Xva_r = sc_r.transform(emb_pool[ext_val])
        Xte_r = sc_r.transform(emb_te)
        r_score = {c: fit_ridge(Xtr_r, y_tr[c], Xva_r, y_va[c], Xte_r) for c in CFGS}
        r_score_mat = np.stack([r_score[c] for c in CFGS], 1)
        d1_r = r_score["50steps"]-r_score["10steps"]
        d2_r = r_score["tts"]-r_score["50steps"]
        r_delta_mat = np.stack([-d1_r, np.zeros(n_te), d2_r], 1)
        rows["ridge_score"].append(route_quality(r_score_mat, true_mat, costs, budget))
        rows["ridge_delta"].append(route_quality(r_delta_mat, true_mat, costs, budget))

        # ── TF MAE（原始 BCE loss，归一化分值）──────────
        tf_mae_score = {}
        for c in CFGS:
            lo, hi = y_tr[c].min(), y_tr[c].max()
            rs = max(hi-lo, 1e-6)
            nf = lambda a, lo=lo, rs=rs: (a-lo)/rs
            pred_n, _, _ = TF.train_tf_mlp(
                Fpool[sub], Mpool[sub], nf(y_tr[c]).reshape(-1,1),
                Fva, Mva, nf(y_va[c]).reshape(-1,1),
                Fte, Mte, M=1, seed=seed, device=dev)
            tf_mae_score[c] = pred_n[:,0]*rs + lo
        tf_mae_score_mat = np.stack([tf_mae_score[c] for c in CFGS], 1)
        rows["tf_mae_score"].append(route_quality(tf_mae_score_mat, true_mat, costs, budget))

        d_va = np.stack([y_va["50steps"]-y_va["10steps"], y_va["tts"]-y_va["50steps"]], 1)
        lo_d, hi_d = d_tr.min(), d_tr.max(); rd = max(hi_d-lo_d, 1e-6)
        nd = lambda a: (a-lo_d)/rd
        pred_dn, _, _ = TF.train_tf_mlp(
            Fpool[sub], Mpool[sub], nd(d_tr),
            Fva, Mva, nd(d_va),
            Fte, Mte, M=2, seed=seed, device=dev)
        tf_mae_delta_raw = pred_dn*rd + lo_d
        tf_mae_delta_mat = np.stack([-tf_mae_delta_raw[:,0], np.zeros(n_te), tf_mae_delta_raw[:,1]], 1)
        rows["tf_mae_delta"].append(route_quality(tf_mae_delta_mat, true_mat, costs, budget))
        print(f"  seed={seed} ridge_s={rows['ridge_score'][-1]:.4f}  "
              f"tf_mae_s={rows['tf_mae_score'][-1]:.4f}  tf_mae_d={rows['tf_mae_delta'][-1]:.4f}",
              end="  ", flush=True)

        # ── TF Soft-DP（score mode：预测 3 绝对分值）───
        y_tr_mat_s = np.stack([y_tr[c] for c in CFGS], 1)  # (n_tr, 3)
        y_va_mat_s = np.stack([y_va[c] for c in CFGS], 1)
        pred_dp_s, _, _ = TF.train_tf_mlp_softdp(
            Fpool[sub], Mpool[sub], y_tr_mat_s,
            Fva, Mva, y_va_mat_s,
            Fte, Mte,
            costs=costs, budget=budget,
            M=3, seed=seed, device=dev,
            max_epochs=200, lr=1e-3, lr_dp=2e-4, batch=128, patience=40,
            tau_start=2.0, tau_end=10.0, loss_mode="exact",
            pretrain_epochs=50, dp_batch=32)
        # pred_dp_s: (n_te, 3) 原始分值（logit，不归一化）
        rows["tf_dp_score"].append(route_quality(pred_dp_s, true_mat, costs, budget))

        # ── TF Soft-DP（delta mode：预测 2 个 delta）───
        # d1 = q50-q10, d2 = qtts-q50；构造相对分矩阵 [-d1, 0, d2]
        y_tr_mat = np.stack([y_tr[c] for c in CFGS], 1)  # (n_tr, 3) 真实质量分
        y_va_mat = np.stack([y_va[c] for c in CFGS], 1)
        pred_dp_d, _, _ = TF.train_tf_mlp_softdp(
            Fpool[sub], Mpool[sub], d_tr,
            Fva, Mva, d_va,
            Fte, Mte,
            costs=costs, budget=budget,
            M=2, seed=seed, device=dev,
            max_epochs=200, lr=1e-3, lr_dp=2e-4, batch=128, patience=40,
            tau_start=2.0, tau_end=10.0, loss_mode="exact",
            delta_mode=True, pretrain_epochs=50, dp_batch=32,
            true_scores_tr=y_tr_mat, true_scores_va=y_va_mat)
        # pred_dp_d: (n_te, 2): [d1_pred, d2_pred]
        dp_delta_mat = np.stack([-pred_dp_d[:,0], np.zeros(n_te), pred_dp_d[:,1]], 1)
        rows["tf_dp_delta"].append(route_quality(dp_delta_mat, true_mat, costs, budget))
        print(f"tf_dp_s={rows['tf_dp_score'][-1]:.4f}  tf_dp_d={rows['tf_dp_delta'][-1]:.4f}",
              flush=True)

    out = {"dataset": dataset, "model": model,
           "oracle": oracle, "lp": lp, "always_50steps": always_50}
    for key, vals in rows.items():
        out[key+"_mean"] = float(np.mean(vals))
        out[key+"_std"]  = float(np.std(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",       type=int, default=0)
    ap.add_argument("--datasets",  nargs="+", default=DATASETS)
    ap.add_argument("--models",    nargs="+", default=MODELS)
    ap.add_argument("--out_suffix",default="")
    a = ap.parse_args()

    save_path = os.path.join(RES, f"exp11_softdp{a.out_suffix}.json")
    if os.path.exists(save_path):
        all_results = json.load(open(save_path))
        done = {(r["dataset"],r["model"]) for r in all_results}
        print(f"[resume] done={done}")
    else:
        all_results = []; done = set()

    for dataset in a.datasets:
        for model in a.models:
            if (dataset,model) in done:
                print(f"[skip] {dataset}/{model}"); continue
            result = run_group(dataset, model, a.gpu)
            all_results.append(result)
            done.add((dataset,model))
            json.dump(all_results, open(save_path,"w"), indent=2)
            print(f"[saved] {save_path}", flush=True)

    print(f"[done] {save_path}")

if __name__ == "__main__":
    main()
