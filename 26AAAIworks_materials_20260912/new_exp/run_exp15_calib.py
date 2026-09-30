#!/usr/bin/env python3
"""
§15 Isotonic Calibration variants + Random Forest: gm/qwen

variants:
  ridge          baseline
  ridge_iso      Ridge + per-config isotonic calibration on val
  ridge_iso_joint Ridge + joint isotonic（对 routing score q_j - λ*·c_j 做校准）
  rf             RandomForest 直接回归 3 个分值
  rf_iso         RF + per-config isotonic calibration
  ridge_rf_ens   Ridge + RF ensemble（平均）
  ridge_rf_iso   Ridge + RF ensemble + isotonic calibration
"""
import os, sys, json, random, argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
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


def route_quality(pred_mat, true_mat):
    bucket = defaultdict(float)
    for lam in LAMBDAS:
        adj = pred_mat - lam * COSTS[None, :]
        chosen = np.argmax(adj, axis=1)
        c = float(np.mean(COSTS[chosen]))
        q = float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))
        bucket[round(c, 6)] = max(bucket[round(c, 6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if BUDGET <= xs[0]:  return float(qs[0])
    if BUDGET >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, BUDGET))
    c0, c1 = xs[idx-1], xs[idx]; q0, q1 = qs[idx-1], qs[idx]
    return float(q0 + (BUDGET - c0) / (c1 - c0) * (q1 - q0))


def fit_ridge_mat(Xtr, Ytr, Xva, Yva, Xte):
    preds = []
    for j in range(Ytr.shape[1]):
        best_mse, best_m = 1e18, None
        for a in [0.1, 1., 10., 100., 1000., 1e4]:
            m = Ridge(alpha=a).fit(Xtr, Ytr[:, j])
            mse = float(np.mean((m.predict(Xva) - Yva[:, j]) ** 2))
            if mse < best_mse: best_mse = mse; best_m = m
        preds.append(best_m.predict(Xte))
    return np.stack(preds, 1)   # (n_te, 3)


def fit_rf_mat(Xtr, Ytr, Xte, seed=0):
    """RandomForest 回归 3 个 config 分值，多输出"""
    rf = RandomForestRegressor(
        n_estimators=200, max_features="sqrt",
        min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, Ytr)
    return rf.predict(Xte)   # (n_te, 3)


def isotonic_calib(pred_va, true_va, pred_te):
    """per-config isotonic calibration: 用 val 上 (pred, true) 对拟合单调映射"""
    out = np.empty_like(pred_te)
    for j in range(pred_va.shape[1]):
        # 按 pred 排序后拟合 isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pred_va[:, j], true_va[:, j])
        out[:, j] = iso.predict(pred_te[:, j])
    return out


def isotonic_calib_joint(pred_va, true_va, pred_te, lam_star):
    """joint: 对 routing score s_j = pred_j - λ*·c_j 做 isotonic 校准"""
    # routing score (N, 3)
    rs_va = pred_va - lam_star * COSTS[None, :]
    rs_te = pred_te - lam_star * COSTS[None, :]
    # 对每个 config 的 routing score 做 isotonic
    out_rs = np.empty_like(rs_te)
    for j in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(rs_va[:, j], true_va[:, j] - lam_star * COSTS[j])
        out_rs[:, j] = iso.predict(rs_te[:, j])
    # 加回 λ*·c_j 还原为伪质量分（不影响 argmax，只影响 Lagrangian 曲线形状）
    return out_rs + lam_star * COSTS[None, :]


def find_lam_star(true_mat):
    best_lam, best_gap = 0.0, 1e9
    for lam in LAMBDAS:
        adj = true_mat - lam * COSTS[None, :]
        avg_c = float(np.mean(COSTS[np.argmax(adj, 1)]))
        gap = abs(avg_c - BUDGET)
        if gap < best_gap: best_gap = gap; best_lam = lam
    return best_lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    print("=== §15 Isotonic Calib + RF: gm/qwen ===")
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

    VARIANTS = ["ridge", "ridge_iso", "ridge_iso_joint",
                "rf", "rf_iso", "ridge_rf_ens", "ridge_rf_iso"]
    results = {v: [] for v in VARIANTS}

    for seed in SEEDS:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(N_TRAIN, len(lc_pool))))
        Ytr_mat = np.stack([pool_scores[c][sub] for c in CFGS], 1)
        lam_star = find_lam_star(Ytr_mat)

        sc = StandardScaler().fit(Xpool[sub])
        Xtr = sc.transform(Xpool[sub])
        Xva = sc.transform(Xva_raw)
        Xte = sc.transform(Xte_np)

        # Ridge 预测（val 和 test）
        pred_va_r = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xva)
        pred_te_r = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xte)

        # RF 预测
        pred_va_f = fit_rf_mat(Xtr, Ytr_mat, Xva, seed=seed)
        pred_te_f = fit_rf_mat(Xtr, Ytr_mat, Xte, seed=seed)

        row = []

        # 1. ridge baseline
        rq = route_quality(pred_te_r, true_mat)
        results["ridge"].append(rq); row.append(f"ridge={rq:.4f}")

        # 2. ridge + per-config isotonic
        pred_te_ri = isotonic_calib(pred_va_r, Yva_mat, pred_te_r)
        rq = route_quality(pred_te_ri, true_mat)
        results["ridge_iso"].append(rq); row.append(f"ridge_iso={rq:.4f}")

        # 3. ridge + joint isotonic
        pred_te_rij = isotonic_calib_joint(pred_va_r, Yva_mat, pred_te_r, lam_star)
        rq = route_quality(pred_te_rij, true_mat)
        results["ridge_iso_joint"].append(rq); row.append(f"ridge_iso_j={rq:.4f}")

        # 4. rf baseline
        rq = route_quality(pred_te_f, true_mat)
        results["rf"].append(rq); row.append(f"rf={rq:.4f}")

        # 5. rf + per-config isotonic
        pred_te_fi = isotonic_calib(pred_va_f, Yva_mat, pred_te_f)
        rq = route_quality(pred_te_fi, true_mat)
        results["rf_iso"].append(rq); row.append(f"rf_iso={rq:.4f}")

        # 6. ridge + rf ensemble
        pred_te_ens = 0.5 * pred_te_r + 0.5 * pred_te_f
        rq = route_quality(pred_te_ens, true_mat)
        results["ridge_rf_ens"].append(rq); row.append(f"ens={rq:.4f}")

        # 7. ridge + rf ensemble + isotonic
        pred_va_ens = 0.5 * pred_va_r + 0.5 * pred_va_f
        pred_te_ensi = isotonic_calib(pred_va_ens, Yva_mat, pred_te_ens)
        rq = route_quality(pred_te_ensi, true_mat)
        results["ridge_rf_iso"].append(rq); row.append(f"ens_iso={rq:.4f}")

        print(f"  seed={seed}  " + "  ".join(row), flush=True)

    print("\n=== 汇总（3 seeds 均值±std）===")
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}")
    baseline = np.mean(results["ridge"])
    for v in VARIANTS:
        vals = results[v]
        print(f"  {v:<18}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"vs_ridge={np.mean(vals)-baseline:+.4f}")

    out = os.path.join(HERE, "results", "exp15_calib.json")
    json.dump({
        "oracle": oracle_q, "always_50steps": always50,
        **{f"{v}_mean": float(np.mean(results[v])) for v in VARIANTS},
        **{f"{v}_std":  float(np.std(results[v]))  for v in VARIANTS},
        "per_seed": {v: [float(x) for x in results[v]] for v in VARIANTS},
    }, open(out, "w"), indent=2)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
