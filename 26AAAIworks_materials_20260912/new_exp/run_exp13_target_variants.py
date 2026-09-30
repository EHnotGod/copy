#!/usr/bin/env python3
"""
§13 Training Target Variants for Routing（gm/qwen, Qwen embedding + Ridge/LogReg）

只改训练目标，特征用 Qwen3-Embedding-8B，predictor 用 Ridge/LogReg。

方案：
  score      : Ridge 预测 3 个绝对分（baseline，同 exp4）
  lam_adj    : Ridge 预测 q_j - λ*·cost_j，λ* 由训练集 oracle 确定，test 时 argmax
  oracle_cls : 训练集 oracle routing 标签（0/1/2）做 3-class LogReg，test 时 argmax prob → Lagrangian
  rank_norm  : Ridge 预测 q_j - mean_query(q)（每条 query 内去均值），test 时 Lagrangian
"""
import os, sys, json, random, argparse
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
import catimage_tf_mlp as TF

DATA  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CLIP_DIR  = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"

CFGS   = ["10steps", "50steps", "tts"]
COSTS  = np.array([2.0, 10.0, 30.5])
BUDGET = 10.0
SEEDS  = [0, 1, 2]
N_TRAIN = 2200
N_VAL   = 200

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000), np.linspace(0.05, 3.0, 700), [10., 1e6]]))


# ── 数据加载 ──────────────────────────────────────────────────
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


# ── Lagrangian routing ────────────────────────────────────────
def route_quality(pred_mat, true_mat):
    pts = []
    for lam in LAMBDAS:
        adj    = pred_mat - lam * COSTS[None, :]
        chosen = np.argmax(adj, axis=1)
        pts.append((float(np.mean(COSTS[chosen])),
                    float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))))
    bucket = defaultdict(float)
    for c, q in pts:
        bucket[round(c, 6)] = max(bucket[round(c, 6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if BUDGET <= xs[0]:  return float(qs[0])
    if BUDGET >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, BUDGET))
    c0, c1 = xs[idx-1], xs[idx]; q0, q1 = qs[idx-1], qs[idx]
    return float(q0 + (BUDGET - c0) / (c1 - c0) * (q1 - q0))


def find_lambda_star(true_mat):
    """在给定 true_mat 上找满足预算约束的最优 λ*"""
    best_lam, best_q, best_cost_gap = None, -1e9, 1e9
    for lam in LAMBDAS:
        adj    = true_mat - lam * COSTS[None, :]
        chosen = np.argmax(adj, axis=1)
        avg_c  = float(np.mean(COSTS[chosen]))
        avg_q  = float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))
        gap    = abs(avg_c - BUDGET)
        if gap < best_cost_gap or (gap == best_cost_gap and avg_q > best_q):
            best_cost_gap = gap; best_q = avg_q; best_lam = lam
    return best_lam


def oracle_labels(true_mat, lam_star):
    """用 λ* 计算每条 query 的 oracle 选择标签"""
    adj = true_mat - lam_star * COSTS[None, :]
    return np.argmax(adj, axis=1).astype(int)


# ── Ridge fit helper ──────────────────────────────────────────
def fit_ridge(Xtr, ytr, Xva, yva, Xte):
    best_sc, best_m = -1e18, None
    for a in [0.1, 1., 10., 100., 1000., 1e4]:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = float(np.mean((m.predict(Xva) - yva) ** 2))   # val MSE
        if -sc > best_sc: best_sc = -sc; best_m = m
    return best_m.predict(Xte)


def fit_logreg(Xtr, ytr, Xva, yva, Xte):
    """多分类 logistic regression，返回 (n_te, 3) 概率矩阵"""
    best_sc, best_m = -1e18, None
    for c in [0.01, 0.1, 1., 10.]:
        m = LogisticRegression(C=c, max_iter=1000,
                               solver="lbfgs").fit(Xtr, ytr)
        sc = float(m.score(Xva, yva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m.predict_proba(Xte)   # (n_te, 3)


# ── 主实验 ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    print("=== §13 Target Variants: gm/qwen ===")
    p1 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen2000_perseed.json"
    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_instrs = i1 + i2
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}

    te_instrs, yte_cfg = load_scores(
        f"{DATA}/geneval2/edit/geneval2_qwen_edit_raw_perseed.json")
    n_te = len(te_instrs)
    true_mat = np.stack([yte_cfg[c] for c in CFGS], axis=1)
    oracle_q  = route_quality(true_mat, true_mat)
    always50  = float(yte_cfg["50steps"].mean())
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}  test={n_te}")

    # CLIP token features → mean pool → (N, 768)
    Fpool, Mpool = TF.clip_token_features(pool_instrs, "exp4_gm_qwen_pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   "exp4_gm_qwen_test", CLIP_DIR, dev)
    def mean_pool(F, M):
        m = M[:, :, None]
        return (F * m).sum(1) / np.clip(m.sum(1), 1e-6, None)
    emb_pool = mean_pool(Fpool, Mpool)   # (N, 768)
    emb_te   = mean_pool(Fte, Mte)
    print(f"  emb pool={emb_pool.shape}  test={emb_te.shape}", flush=True)

    rng_fix = random.Random(42)
    all_idx  = list(range(len(pool_instrs))); rng_fix.shuffle(all_idx)
    ext_val  = np.array(all_idx[:N_VAL]); lc_pool = all_idx[N_VAL:]
    y_va     = {c: pool_scores[c][ext_val] for c in CFGS}
    Yva_mat  = np.stack([y_va[c] for c in CFGS], 1)

    VARIANTS = ["score", "lam_adj", "oracle_cls", "rank_norm"]
    results  = {v: [] for v in VARIANTS}

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        y_tr    = {c: pool_scores[c][sub] for c in CFGS}
        Ytr_mat = np.stack([y_tr[c] for c in CFGS], 1)   # (n_tr, 3)

        sc = StandardScaler().fit(emb_pool[sub])
        Xtr = sc.transform(emb_pool[sub])
        Xva = sc.transform(emb_pool[ext_val])
        Xte = sc.transform(emb_te)

        row = []

        # ── 1. score（baseline）──────────────────────────
        pred_s = np.stack([fit_ridge(Xtr, y_tr[c], Xva, y_va[c], Xte)
                           for c in CFGS], axis=1)
        rq = route_quality(pred_s, true_mat)
        results["score"].append(rq); row.append(f"score={rq:.4f}")

        # ── 2. λ*-adjusted regression ─────────────────────
        lam_star = find_lambda_star(Ytr_mat)
        # target_j = q_j - λ* · cost_j（折入 cost，让 Ridge 直接学路由价值）
        Ytr_adj = Ytr_mat - lam_star * COSTS[None, :]   # (n_tr, 3)
        Yva_adj = Yva_mat - lam_star * COSTS[None, :]
        pred_adj = np.stack([
            fit_ridge(Xtr, Ytr_adj[:, j], Xva, Yva_adj[:, j], Xte)
            for j in range(3)], axis=1)
        rq = route_quality(pred_adj, true_mat)
        results["lam_adj"].append(rq); row.append(f"lam_adj={rq:.4f}(λ*={lam_star:.4f})")

        # ── 3. oracle 分类（LogReg on 0/1/2 labels）───────
        labels_tr = oracle_labels(Ytr_mat, lam_star)
        labels_va = oracle_labels(Yva_mat, lam_star)
        # 检查类别是否足够（若 tts 从未被选中，退化为 2-class）
        unique_cls = np.unique(labels_tr)
        if len(unique_cls) < 2:
            # 所有样本同一类，退化到 score baseline
            results["oracle_cls"].append(results["score"][-1])
            row.append(f"oracle_cls=degenerate")
        else:
            prob_te = fit_logreg(Xtr, labels_tr, Xva, labels_va, Xte)  # (n_te, K)
            # 如果 LogReg 类别少于 3，补齐到 3 列（缺失类概率=0）
            if prob_te.shape[1] < 3:
                full = np.zeros((n_te, 3))
                for ki, cls in enumerate(unique_cls): full[:, cls] = prob_te[:, ki]
                prob_te = full
            rq = route_quality(prob_te, true_mat)
            results["oracle_cls"].append(rq); row.append(f"oracle_cls={rq:.4f}")

        # ── 4. rank_norm（去均值，学 query 内相对排序）────
        Ytr_norm = Ytr_mat - Ytr_mat.mean(axis=1, keepdims=True)
        Yva_norm = Yva_mat - Yva_mat.mean(axis=1, keepdims=True)
        pred_rn = np.stack([
            fit_ridge(Xtr, Ytr_norm[:, j], Xva, Yva_norm[:, j], Xte)
            for j in range(3)], axis=1)
        rq = route_quality(pred_rn, true_mat)
        results["rank_norm"].append(rq); row.append(f"rank_norm={rq:.4f}")

        print(f"  seed={seed}  " + "  ".join(row), flush=True)

    print("\n=== 汇总（3 seeds 均值±std）===")
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}")
    for v in VARIANTS:
        vals = results[v]
        print(f"  {v:<12}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"vs_score={np.mean(vals)-np.mean(results['score']):+.4f}")

    out = os.path.join(HERE, "results", "exp13_target_variants.json")
    json.dump({
        "dataset": "gm", "model": "qwen",
        "oracle": oracle_q, "always_50steps": always50,
        **{f"{v}_mean": float(np.mean(results[v])) for v in VARIANTS},
        **{f"{v}_std":  float(np.std(results[v]))  for v in VARIANTS},
        "per_seed": {v: [float(x) for x in results[v]] for v in VARIANTS},
    }, open(out, "w"), indent=2)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
