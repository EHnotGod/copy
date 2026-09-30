#!/usr/bin/env python3
"""
§17 DP Routing × Prediction Target: gm/qwen

路由：MCKP 背包 DP（不用 λ，不用 Lagrangian）
预测目标：多种组合
  score         : Ridge 预测 3 个绝对分（baseline）
  delta         : Ridge 预测 [q50-q10, qtts-q50]，DP score = [-d1, 0, d2]
  rank_norm     : Ridge 预测 q_j - mean_q（去均值）
  rank_zscore   : Ridge 预测 (q_j - mean_q) / std_q（归一化到方差=1）
  delta_cn      : Ridge 预测 delta / delta_cost（单位成本边际增益）
  percentile    : 把预测值转为 per-config percentile rank（0~1），消除均值偏移
  rank_norm_var : rank_norm + variance inflation（撑回训练集真实 std）

评估：DP routing quality（直接用 true score 回溯验证选了哪些）
同时报告：oracle DP / 50steps baseline
"""
import os, sys, json, random, argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
import catimage_tf_mlp as TF
sys.path.insert(0, HERE)
from run_exp10_dp_routing import knapsack_dp_route

DATA     = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CLIP_DIR = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"
CFGS     = ["10steps", "50steps", "tts"]
COSTS    = np.array([2.0, 10.0, 30.5])
BUDGET   = 10.0
INT_COSTS = np.round(COSTS * 10).astype(int)   # [20, 100, 305]
N_TRAIN  = 2200
N_VAL    = 200
SEEDS    = [0, 1, 2]


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


def dp_rq(pred_mat, true_mat):
    """MCKP 背包 DP 路由质量（不用 λ）"""
    return knapsack_dp_route(pred_mat, true_mat, INT_COSTS, BUDGET)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    print("=== §17 DP Routing × Prediction Target: gm/qwen ===")
    p1 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen2000_perseed.json"
    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_instrs = i1 + i2
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}
    te_instrs, yte_cfg = load_scores(
        f"{DATA}/geneval2/edit/geneval2_qwen_edit_raw_perseed.json")
    n_te = len(te_instrs)
    true_mat = np.stack([yte_cfg[c] for c in CFGS], axis=1)

    oracle_dp = dp_rq(true_mat, true_mat)
    always50  = float(yte_cfg["50steps"].mean())
    print(f"  oracle_dp={oracle_dp:.4f}  50steps={always50:.4f}  n_te={n_te}")

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

    VARIANTS = ["score", "delta", "rank_norm", "rank_zscore",
                "delta_cn", "percentile", "rank_norm_var"]
    results = {v: [] for v in VARIANTS}

    dc_10_50  = COSTS[1] - COSTS[0]   # 8.0
    dc_50_tts = COSTS[2] - COSTS[1]   # 20.5

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        Ytr_mat = np.stack([pool_scores[c][sub] for c in CFGS], 1)
        sc_s = StandardScaler().fit(Xpool[sub])
        Xtr = sc_s.transform(Xpool[sub])
        Xva = sc_s.transform(Xva_raw)
        Xte = sc_s.transform(Xte_np)

        # ── 1. score（baseline）──────────────────────────────
        P = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xte)
        results["score"].append(dp_rq(P, true_mat))

        # ── 2. delta ─────────────────────────────────────────
        Ytr_d = np.stack([Ytr_mat[:,1]-Ytr_mat[:,0],
                          Ytr_mat[:,2]-Ytr_mat[:,1]], 1)
        Yva_d = np.stack([Yva_mat[:,1]-Yva_mat[:,0],
                          Yva_mat[:,2]-Yva_mat[:,1]], 1)
        P_d = fit_ridge_mat(Xtr, Ytr_d, Xva, Yva_d, Xte)
        # DP score: [-d1, 0, d2]（从 10steps 出发的相对增量）
        P_dp = np.stack([-P_d[:,0], np.zeros(n_te), P_d[:,1]], 1)
        results["delta"].append(dp_rq(P_dp, true_mat))

        # ── 3. rank_norm ──────────────────────────────────────
        mu_tr = Ytr_mat.mean(1, keepdims=True)
        mu_va = Yva_mat.mean(1, keepdims=True)
        P_rn = fit_ridge_mat(Xtr, Ytr_mat-mu_tr, Xva, Yva_mat-mu_va, Xte)
        results["rank_norm"].append(dp_rq(P_rn, true_mat))

        # ── 4. rank_zscore ────────────────────────────────────
        std_tr = (Ytr_mat - mu_tr).std(1, keepdims=True) + 1e-6
        std_va = (Yva_mat - mu_va).std(1, keepdims=True) + 1e-6
        P_rz = fit_ridge_mat(Xtr, (Ytr_mat-mu_tr)/std_tr,
                             Xva, (Yva_mat-mu_va)/std_va, Xte)
        results["rank_zscore"].append(dp_rq(P_rz, true_mat))

        # ── 5. delta_costnorm ─────────────────────────────────
        Ytr_cn = np.stack([Ytr_d[:,0]/dc_10_50,
                           Ytr_d[:,1]/dc_50_tts], 1)
        Yva_cn = np.stack([Yva_d[:,0]/dc_10_50,
                           Yva_d[:,1]/dc_50_tts], 1)
        P_cn = fit_ridge_mat(Xtr, Ytr_cn, Xva, Yva_cn, Xte)
        P_cn_dp = np.stack([-P_cn[:,0]*dc_10_50, np.zeros(n_te),
                             P_cn[:,1]*dc_50_tts], 1)
        results["delta_cn"].append(dp_rq(P_cn_dp, true_mat))

        # ── 6. percentile（per-config 转 percentile rank）──────
        # 消除每个 config 预测值的绝对偏移，只保留排名信息
        P_abs = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xte)
        P_pct = np.empty_like(P_abs)
        for j in range(3):
            # 在训练集预测值的经验分布上查 percentile
            tr_pred_j = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xtr)[:, j]
            ranks = np.searchsorted(np.sort(tr_pred_j), P_abs[:, j])
            P_pct[:, j] = ranks / len(tr_pred_j)
        results["percentile"].append(dp_rq(P_pct, true_mat))

        # ── 7. rank_norm + variance inflation ─────────────────
        # 先在训练集 in-sample 预测，算 std 比值
        P_tr_rn = fit_ridge_mat(Xtr, Ytr_mat-mu_tr, Xva, Yva_mat-mu_va, Xtr)
        P_rn_vi = P_rn.copy()
        for j in range(3):
            std_y = float(np.std(Ytr_mat[:, j] - mu_tr[:, 0]))
            std_p = float(np.std(P_tr_rn[:, j])) + 1e-9
            k = std_y / std_p
            mean_p = float(np.mean(P_tr_rn[:, j]))
            P_rn_vi[:, j] = mean_p + (P_rn[:, j] - mean_p) * k
        results["rank_norm_var"].append(dp_rq(P_rn_vi, true_mat))

        row = [f"{v}={results[v][-1]:.4f}" for v in VARIANTS]
        print(f"  seed={seed}  " + "  ".join(row), flush=True)

    print("\n=== 汇总（3 seeds 均值±std）===")
    print(f"  oracle_dp={oracle_dp:.4f}  50steps={always50:.4f}")
    baseline = np.mean(results["score"])
    for v in VARIANTS:
        vals = results[v]
        print(f"  {v:<16}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"vs_score={np.mean(vals)-baseline:+.4f}")

    out = os.path.join(HERE, "results", "exp17_dp_targets.json")
    json.dump({
        "oracle_dp": oracle_dp, "always_50steps": always50,
        **{f"{v}_mean": float(np.mean(results[v])) for v in VARIANTS},
        **{f"{v}_std":  float(np.std(results[v]))  for v in VARIANTS},
        "per_seed": {v: [float(x) for x in results[v]] for v in VARIANTS},
    }, open(out, "w"), indent=2)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
