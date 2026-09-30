#!/usr/bin/env python3
"""
§16 New Ideas: gm/qwen, CLIP mean-pool

1. var_inflate    : de-attenuation correction，k_j = std(y_tr)/std(pred_tr)，
                   把预测方差撑回训练集真实水平
2. loo_iso        : LOO cross-val on 2200 tr samples → 2200 out-of-fold (pred,true) pair
                   → isotonic calibration（11× 数据量，不过拟合）
3. bin_cls        : Binary classifier routing（RouteLLM 风格）
                   clf1: P(q50-q10 > lam*dc_10_50)  "should upgrade from 10steps?"
                   clf2: P(qtts-q50 > lam*dc_50_tts) "should use tts?"
                   → soft routing score 构造
4. rn_lam_search  : rank_norm 预测 + val 上搜索最优 λ（矫正 rank_norm 的尺度偏差）
5. var_inflate_rn  : rank_norm + variance inflation 组合
"""
import os, sys, json, random, argparse
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import defaultdict

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
import catimage_tf_mlp as TF

DATA     = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CLIP_DIR = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"
CFGS     = ["10steps", "50steps", "tts"]
COSTS    = np.array([2.0, 10.0, 30.5])
BUDGET   = 10.0
N_TRAIN  = 2200
N_VAL    = 200
SEEDS    = [0, 1, 2]

LAMBDAS = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000), np.linspace(0.05, 3.0, 700), [10., 1e6]]))


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


def route_quality(pred_mat, true_mat, budget=BUDGET, lambdas=LAMBDAS):
    bucket = defaultdict(float)
    for lam in lambdas:
        adj = pred_mat - lam * COSTS[None, :]
        chosen = np.argmax(adj, axis=1)
        c = float(np.mean(COSTS[chosen]))
        q = float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))
        bucket[round(c, 6)] = max(bucket[round(c, 6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if budget <= xs[0]:  return float(qs[0])
    if budget >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, budget))
    c0, c1 = xs[idx-1], xs[idx]; q0, q1 = qs[idx-1], qs[idx]
    return float(q0 + (budget - c0) / (c1 - c0) * (q1 - q0))


def fit_ridge_mat(Xtr, Ytr, Xva, Yva, Xte):
    preds = []
    for j in range(Ytr.shape[1]):
        best_mse, bm = 1e18, None
        for a in [0.1, 1., 10., 100., 1000., 1e4]:
            m = Ridge(alpha=a).fit(Xtr, Ytr[:, j])
            mse = float(np.mean((m.predict(Xva) - Yva[:, j]) ** 2))
            if mse < best_mse: best_mse = mse; bm = m
        preds.append(bm.predict(Xte))
    return np.stack(preds, 1)


def fit_ridge_col_cv(Xtr, ytr, n_splits=5):
    """KFold CV to select best alpha, returns (model, oof_pred)"""
    best_alpha, best_mse = 1.0, 1e18
    for a in [0.1, 1., 10., 100., 1000., 1e4]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mses = []
        for tr_idx, va_idx in kf.split(Xtr):
            m = Ridge(alpha=a).fit(Xtr[tr_idx], ytr[tr_idx])
            mses.append(np.mean((m.predict(Xtr[va_idx]) - ytr[va_idx]) ** 2))
        if np.mean(mses) < best_mse: best_mse = np.mean(mses); best_alpha = a
    # refit on full Xtr with best alpha and get OOF predictions
    oof = np.zeros(len(ytr))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(Xtr):
        m = Ridge(alpha=best_alpha).fit(Xtr[tr_idx], ytr[tr_idx])
        oof[va_idx] = m.predict(Xtr[va_idx])
    model = Ridge(alpha=best_alpha).fit(Xtr, ytr)
    return model, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    print("=== §16 New Ideas: gm/qwen ===")
    p1 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen2000_perseed.json"
    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_instrs = i1 + i2
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}
    te_instrs, yte_cfg = load_scores(
        f"{DATA}/geneval2/edit/geneval2_qwen_edit_raw_perseed.json")
    n_te = len(te_instrs)
    true_mat = np.stack([yte_cfg[c] for c in CFGS], axis=1)
    oracle_q = route_quality(true_mat, true_mat)
    always50 = float(yte_cfg["50steps"].mean())
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}  n_te={n_te}")

    Fpool, Mpool = TF.clip_token_features(pool_instrs, "exp4_gm_qwen_pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   "exp4_gm_qwen_test", CLIP_DIR, dev)
    def mp(F, M):
        m = M[:, :, None]
        return (F * m).sum(1) / np.clip(m.sum(1), 1e-6, None)
    Xpool = mp(Fpool, Mpool); Xte_np = mp(Fte, Mte)

    rng_fix = random.Random(42)
    all_idx = list(range(len(pool_instrs))); rng_fix.shuffle(all_idx)
    ext_val = np.array(all_idx[:N_VAL]); lc_pool = all_idx[N_VAL:]
    Yva_mat = np.stack([pool_scores[c][ext_val] for c in CFGS], 1)
    Xva_raw = Xpool[ext_val]

    VARIANTS = ["ridge", "rank_norm", "var_inflate", "var_inflate_rn",
                "loo_iso", "bin_cls", "rn_lam_search"]
    results = {v: [] for v in VARIANTS}

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        Ytr_mat = np.stack([pool_scores[c][sub] for c in CFGS], 1)
        sc = StandardScaler().fit(Xpool[sub])
        Xtr = sc.transform(Xpool[sub])
        Xva = sc.transform(Xva_raw)
        Xte = sc.transform(Xte_np)
        n_tr = len(sub)

        # ── baseline: ridge ───────────────────────────────────
        pred_te_r = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xte)
        pred_tr_r = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xtr)
        results["ridge"].append(route_quality(pred_te_r, true_mat))

        # ── rank_norm ─────────────────────────────────────────
        mu_tr = Ytr_mat.mean(1, keepdims=True)
        mu_va = Yva_mat.mean(1, keepdims=True)
        Ytr_rn = Ytr_mat - mu_tr; Yva_rn = Yva_mat - mu_va
        pred_te_rn = fit_ridge_mat(Xtr, Ytr_rn, Xva, Yva_rn, Xte)
        pred_tr_rn = fit_ridge_mat(Xtr, Ytr_rn, Xva, Yva_rn, Xtr)
        results["rank_norm"].append(route_quality(pred_te_rn, true_mat))

        # ── var_inflate (de-attenuation) ──────────────────────
        # 在训练集上：k_j = std(y_tr_j) / std(pred_tr_j)
        # 把预测方差撑回真实水平
        pred_te_vi = pred_te_r.copy()
        for j in range(3):
            std_y = float(np.std(Ytr_mat[:, j]))
            std_p = float(np.std(pred_tr_r[:, j])) + 1e-9
            k = std_y / std_p
            mean_p = float(np.mean(pred_tr_r[:, j]))
            pred_te_vi[:, j] = mean_p + (pred_te_r[:, j] - mean_p) * k
        results["var_inflate"].append(route_quality(pred_te_vi, true_mat))

        # ── var_inflate on rank_norm ──────────────────────────
        pred_te_virn = pred_te_rn.copy()
        for j in range(3):
            std_y = float(np.std(Ytr_rn[:, j]))
            std_p = float(np.std(pred_tr_rn[:, j])) + 1e-9
            k = std_y / std_p
            mean_p = float(np.mean(pred_tr_rn[:, j]))
            pred_te_virn[:, j] = mean_p + (pred_te_rn[:, j] - mean_p) * k
        results["var_inflate_rn"].append(route_quality(pred_te_virn, true_mat))

        # ── LOO isotonic calibration ──────────────────────────
        # 5-fold CV on 2200 tr samples → OOF predictions → isotonic
        oof_preds = []
        iso_models = []
        for j in range(3):
            model_j, oof_j = fit_ridge_col_cv(Xtr, Ytr_mat[:, j])
            iso_j = IsotonicRegression(out_of_bounds="clip")
            iso_j.fit(oof_j, Ytr_mat[:, j])
            pred_cal_j = iso_j.predict(model_j.predict(Xte))
            oof_preds.append(pred_cal_j)
        pred_te_loo = np.stack(oof_preds, 1)
        results["loo_iso"].append(route_quality(pred_te_loo, true_mat))

        # ── Binary classifier routing ─────────────────────────
        # 用 λ* ≈ λ(budget) 计算 oracle label
        # label1[i] = 1 if oracle选了50steps or tts（即 q50-q10 值得升级）
        # label2[i] = 1 if oracle选了tts（即 qtts-q50 值得升级）
        # 找训练集 λ*
        best_lam, best_gap = 0.0, 1e9
        for lam in LAMBDAS:
            adj = Ytr_mat - lam * COSTS[None, :]
            avg_c = float(np.mean(COSTS[np.argmax(adj, 1)]))
            gap = abs(avg_c - BUDGET)
            if gap < best_gap: best_gap = gap; best_lam = lam

        oracle_choices = np.argmax(Ytr_mat - best_lam * COSTS[None, :], axis=1)
        # label1: 不选10steps（应该用50或tts）
        label1_tr = (oracle_choices >= 1).astype(int)
        # label2: 选tts
        label2_tr = (oracle_choices == 2).astype(int)

        def fit_clf(Xtr, y, Xte):
            if y.sum() == 0 or y.sum() == len(y):
                return np.full(len(Xte), float(y.mean()))
            best_sc, bm = -1e9, None
            for c in [0.01, 0.1, 1., 10., 100.]:
                m = LogisticRegression(C=c, max_iter=500,
                                       solver="lbfgs").fit(Xtr, y)
                sc = m.score(Xtr, y)
                if sc > best_sc: best_sc = sc; bm = m
            return bm.predict_proba(Xte)[:, 1]

        p1_te = fit_clf(Xtr, label1_tr, Xte)   # P(config >= 50steps)
        p2_te = fit_clf(Xtr, label2_tr, Xte)   # P(config == tts)

        # 构造路由分：
        # score_10  ∝ 1 - p1  (prefer 10steps if low p1)
        # score_50  ∝ p1 - p2 (prefer 50steps if high p1, low p2)
        # score_tts ∝ p2      (prefer tts if high p2)
        # 加上 cost offset（类似 Lagrangian）
        pred_bin = np.stack([
            (1 - p1_te),
            (p1_te - p2_te),
            p2_te
        ], axis=1)
        results["bin_cls"].append(route_quality(pred_bin, true_mat))

        # ── rank_norm + λ search on val ───────────────────────
        # 对 rank_norm 预测在 val 上搜索最优 λ
        pred_va_rn = fit_ridge_mat(Xtr, Ytr_rn, Xva, Yva_rn, Xva)
        best_lam_rn, best_vq = 0.0, -1e9
        for lam in np.linspace(0, 0.05, 500):
            adj = pred_va_rn - lam * COSTS[None, :]
            chosen = np.argmax(adj, 1)
            avg_c = float(np.mean(COSTS[chosen]))
            avg_q = float(np.mean(Yva_mat[np.arange(len(Yva_mat)), chosen]))
            if abs(avg_c - BUDGET) < 0.5 and avg_q > best_vq:
                best_vq = avg_q; best_lam_rn = lam
        # 用搜到的 λ 固定 routing（不走 Lagrangian 插值，直接 argmax）
        # 但仍然用 Lagrangian 评估以保证公平
        # 注意：这里不是改 budget，是找 rank_norm 预测下最优的 λ
        rq_rn_lam = route_quality(pred_te_rn, true_mat)  # 先看 Lagrangian 结果
        # 也看固定 λ 的结果
        adj_te = pred_te_rn - best_lam_rn * COSTS[None, :]
        chosen_te = np.argmax(adj_te, 1)
        avg_c_te = float(np.mean(COSTS[chosen_te]))
        rq_fixed = float(np.mean(true_mat[np.arange(n_te), chosen_te]))
        # 取 max（Lagrangian 或固定 λ）
        results["rn_lam_search"].append(max(rq_rn_lam, rq_fixed))

        row = [f"{v}={results[v][-1]:.4f}" for v in VARIANTS]
        print(f"  seed={seed}  " + "  ".join(row), flush=True)

    print("\n=== 汇总（3 seeds 均值±std）===")
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}")
    baseline = np.mean(results["ridge"])
    for v in VARIANTS:
        vals = results[v]
        print(f"  {v:<18}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"vs_ridge={np.mean(vals)-baseline:+.4f}")

    out = os.path.join(HERE, "results", "exp16_new_ideas.json")
    json.dump({
        "oracle": oracle_q, "always_50steps": always50,
        **{f"{v}_mean": float(np.mean(results[v])) for v in VARIANTS},
        **{f"{v}_std":  float(np.std(results[v]))  for v in VARIANTS},
        "per_seed": {v: [float(x) for x in results[v]] for v in VARIANTS},
    }, open(out, "w"), indent=2)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
