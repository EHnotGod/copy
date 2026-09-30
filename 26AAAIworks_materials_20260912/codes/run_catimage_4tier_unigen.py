#!/usr/bin/env python3
"""4 档路由实验 — UniGenBench 版（qwen + flux）。

4 档：qwen-10 / qwen-50 / flux-10 / flux-50
成本：10 / 50 / 23.1 / 115.5
预算：BUDGET_EVAL=50
scores 为直接 float（非 perseed avg），单次打分
Predictors：KNN + TFMLP + RIDGE + LP + OC
train 400 / 800 / 1600 / 2400 对比

注：boogu 的 cost 未知，不纳入路由对比。
"""
import os, sys, json, random, argparse
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
import sklearn.neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catimage_tf_mlp as TF
from embed_lr_dp import load_cached_embedding

UNIGEN_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results"
CACHE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_embed")
MODEL_PATH  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B"

COSTS       = np.array([10.0, 50.0, 23.1, 115.5])
ROUTE_KEYS  = [("qwen","10steps"),("qwen","50steps"),
               ("flux","10steps"),("flux","50steps")]
M           = len(ROUTE_KEYS)
BUDGET_EVAL = 50.0

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catimage_plots_4tier_unigen")
PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catimage_preds_4tier_unigen")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ── 路径 ──────────────────────────────────────────────────────────────
def raw_path(model):
    return f"{UNIGEN_BASE}/unigen_{model}/eval/raw600_edit_v2.json"

def gen_path(model):
    return f"{UNIGEN_BASE}/unigen_{model}/eval/gen2400_edit_v2.json"


# ── 数据加载 ──────────────────────────────────────────────────────────
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _get_score(scores, cfg):
    v = scores.get(cfg)
    if v is None: return None
    return float(v)

def load_raw_merged():
    qraw = load_json(raw_path("qwen"))
    fraw = load_json(raw_path("flux"))
    fidx = {x["instruction"]: x["scores"] for x in fraw}
    out = []
    for q in qraw:
        instr = q["instruction"]
        if instr not in fidx: continue
        qs = q["scores"]; fs = fidx[instr]
        row = [_get_score(qs,"10steps"), _get_score(qs,"50steps"),
               _get_score(fs,"10steps"), _get_score(fs,"50steps")]
        if any(v is None for v in row): continue
        out.append({"instruction": instr, "scores_4": row})
    return out

def load_gen_pool(size):
    qgen = load_json(gen_path("qwen"))
    fgen = load_json(gen_path("flux"))
    fidx = {x["instruction"]: x["scores"] for x in fgen}
    out = []
    for q in qgen:
        instr = q["instruction"]
        if instr not in fidx: continue
        qs = q["scores"]; fs = fidx[instr]
        row = [_get_score(qs,"10steps"), _get_score(qs,"50steps"),
               _get_score(fs,"10steps"), _get_score(fs,"50steps")]
        if any(v is None for v in row): continue
        out.append({"instruction": instr, "scores_4": row})
    return out[:size]


# ── sweep / budget 工具 ───────────────────────────────────────────────
def sweep_tradeoff_curve(gt_scores, route_scores, costs, lambdas):
    n = route_scores.shape[0]
    tuples = []
    for lamb in lambdas:
        ri = np.argmax(route_scores - lamb * costs.reshape(1, -1), axis=1)
        tuples.append((float(np.mean(costs[ri])), float(np.mean(gt_scores[np.arange(n), ri]))))
    tuples.sort(key=lambda x: x[0])
    return np.array([t[0] for t in tuples]), np.array([t[1] for t in tuples])

def quality_at_cost(costs, rewards, budget):
    costs = np.asarray(costs); rewards = np.asarray(rewards)
    inds = np.argsort(costs)
    return float(np.interp(budget, costs[inds], rewards[inds]))

def route_choice_at_budget(route_scores, costs, lambdas, budget):
    costs = np.asarray(costs)
    best = None; best_gap = np.inf
    for lamb in lambdas:
        ri = np.argmax(route_scores - lamb * costs.reshape(1, -1), axis=1)
        gap = abs(float(np.mean(costs[ri])) - budget)
        if gap < best_gap: best_gap = gap; best = ri
    return best


# ── Qwen3 embedding ───────────────────────────────────────────────────
def _load_instr_list(path):
    return [x["instruction"] for x in load_json(path)]

def load_qwen3_emb(model, split):
    p = raw_path(model) if split == "raw" else gen_path(model)
    emb = load_cached_embedding(CACHE_DIR, p, MODEL_PATH)
    if emb is None:
        raise RuntimeError(f"cache miss: {p}")
    return emb

def build_qwen3_features(size, train_data, test_data):
    def _dict(model, split):
        p = raw_path(model) if split == "raw" else gen_path(model)
        emb = load_qwen3_emb(model, split)
        instrs = _load_instr_list(p)
        return {s: emb[i] for i, s in enumerate(instrs)}

    dqg = _dict("qwen", "gen"); dfg = _dict("flux", "gen")
    instr2emb = {}
    for instr in set(dqg) & set(dfg):
        instr2emb[instr] = np.concatenate([dqg[instr], dfg[instr]])

    dqr = _dict("qwen", "raw"); dfr = _dict("flux", "raw")
    X_test = np.array([
        np.concatenate([dqr[x["instruction"]], dfr[x["instruction"]]])
        for x in test_data
    ])
    return instr2emb, X_test

def fit_ridge(Xtr, ytr, Xva, yva, Xte):
    best_sc, best_m = -1e18, None
    for a in [1., 10., 100., 1000.]:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        sc = r2_score(yva, m.predict(Xva))
        if sc > best_sc: best_sc = sc; best_m = m
    return best_m.predict(Xte)


# ── 主实验 ────────────────────────────────────────────────────────────
def run_group(size, seeds, gpu):
    dev = f"cuda:{gpu}"
    print(f"\n===== UniGen 4-tier  train={size}  budget={BUDGET_EVAL} =====", flush=True)

    test_data  = load_raw_merged()
    train_data = load_gen_pool(size)
    n_test = len(test_data)
    print(f"  test={n_test}  train={len(train_data)}", flush=True)

    y_test      = np.array([x["scores_4"] for x in test_data],  dtype=np.float32)
    y_train_all = np.array([x["scores_4"] for x in train_data], dtype=np.float32)

    n_val   = max(50, size // 5)
    n_train = size - n_val

    lambdas = np.concatenate([np.linspace(-0.005, 0.1, 8000), [0.5, 1, 100, -1]])
    oc, oq        = sweep_tradeoff_curve(y_test, y_test, COSTS, lambdas)
    audc_oracle   = quality_at_cost(oc, oq, BUDGET_EVAL)
    oracle_choice = route_choice_at_budget(y_test, COSTS, lambdas, BUDGET_EVAL)

    # CLIP features
    pfx = f"unigen_{size}_"
    def clip_feats(items, tag):
        return TF.clip_token_features([x["instruction"] for x in items], tag, device=dev)
    Ftst, Mtst     = clip_feats(test_data,  pfx+"test")
    Ftrain, Mtrain = clip_feats(train_data, pfx+"train")

    # Qwen3 embedding
    instr2emb, X_test_q3 = build_qwen3_features(size, train_data, test_data)

    audc = {k: [] for k in ["KNN","TFMLP","RIDGE","LP","OC"]}
    acc  = {k: [] for k in ["KNN","TFMLP","RIDGE","LP","OC"]}
    curves = {}; detail = None

    for seed in seeds:
        rng = random.Random(seed)
        idx = rng.sample(range(len(train_data)), min(n_train+n_val, len(train_data)))
        tr, va = idx[:n_train], idx[n_train:n_train+n_val]
        ytr = y_train_all[tr]; yva = y_train_all[va]

        # KNN
        def clip_pool(F, Msk):
            m = Msk[:,:,None]
            return (F*m).sum(1) / np.clip(m.sum(1), 1e-6, None)
        Ctr = clip_pool(Ftrain[tr], Mtrain[tr])
        Cte = clip_pool(Ftst, Mtst)
        sc_ = StandardScaler(); Ctr_s = sc_.fit_transform(Ctr); Cte_s = sc_.transform(Cte)
        k = min(50, len(tr)//3)
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k).fit(Ctr_s)
        idx_nn = nn.kneighbors(Cte_s, return_distance=False)
        rs_knn = np.stack([ytr[idx_nn[i]].mean(0) for i in range(n_test)])
        ck, qk = sweep_tradeoff_curve(y_test, rs_knn, COSTS, lambdas)
        audc["KNN"].append(quality_at_cost(ck, qk, BUDGET_EVAL))
        acc["KNN"].append(float(np.mean(route_choice_at_budget(rs_knn, COSTS, lambdas, BUDGET_EVAL) == oracle_choice)))

        # TFMLP
        lo, hi = ytr.min(), ytr.max(); rng_s = max(hi-lo, 1e-6)
        norm = lambda a: (a-lo)/rng_s
        pred_tf_n, *_ = TF.train_tf_mlp(
            Ftrain[tr], Mtrain[tr], norm(ytr),
            Ftrain[va], Mtrain[va], norm(yva),
            Ftst, Mtst, M=M, seed=seed, device=dev)
        rs_tf = pred_tf_n
        ct, qt = sweep_tradeoff_curve(y_test, rs_tf, COSTS, lambdas)
        audc["TFMLP"].append(quality_at_cost(ct, qt, BUDGET_EVAL))
        acc["TFMLP"].append(float(np.mean(route_choice_at_budget(rs_tf, COSTS, lambdas, BUDGET_EVAL) == oracle_choice)))

        # RIDGE
        Xtr_q3 = np.array([instr2emb[train_data[i]["instruction"]] for i in tr])
        Xva_q3 = np.array([instr2emb[train_data[i]["instruction"]] for i in va])
        sc_q3  = StandardScaler()
        Xtr_q3s = sc_q3.fit_transform(Xtr_q3)
        Xva_q3s = sc_q3.transform(Xva_q3)
        Xte_q3s = sc_q3.transform(X_test_q3)
        rs_rg = fit_ridge(Xtr_q3s, ytr, Xva_q3s, yva, Xte_q3s)
        cr, qr = sweep_tradeoff_curve(y_test, rs_rg, COSTS, lambdas)
        audc["RIDGE"].append(quality_at_cost(cr, qr, BUDGET_EVAL))
        acc["RIDGE"].append(float(np.mean(route_choice_at_budget(rs_rg, COSTS, lambdas, BUDGET_EVAL) == oracle_choice)))

        # LP
        mu_tr = y_train_all.mean(axis=0)
        order_lp = np.argsort(COSTS); cs = COSTS[order_lp]; ms = mu_tr[order_lp]
        hull = [0]
        for i in range(1, M):
            while len(hull)>=2:
                a, b = hull[-2], hull[-1]
                if (ms[i]-ms[a])/(cs[i]-cs[a]+1e-12) >= (ms[b]-ms[a])/(cs[b]-cs[a]+1e-12):
                    hull.pop()
                else: break
            hull.append(i)
        hc = cs[hull]; hm = ms[hull]
        lo_h = hi_h = hull[-1]
        for hi_ in range(1, len(hull)):
            if hc[hi_] >= BUDGET_EVAL: lo_h, hi_h = hull[hi_-1], hull[hi_]; break
        lo_r = order_lp[lo_h]; hi_r = order_lp[hi_h]
        alpha_lp = 1.0 if abs(COSTS[hi_r]-COSTS[lo_r])<1e-9 else float(
            np.clip((BUDGET_EVAL-COSTS[lo_r])/(COSTS[hi_r]-COSTS[lo_r]), 0, 1))
        assign_lp = np.random.default_rng(seed).choice([lo_r,hi_r], size=n_test, p=[1-alpha_lp,alpha_lp])
        avg_q_lp = float(np.mean(y_test[np.arange(n_test), assign_lp]))
        avg_c_lp = float(np.mean(COSTS[assign_lp]))
        audc["LP"].append(avg_q_lp)
        acc["LP"].append(float(np.mean(assign_lp == oracle_choice)))

        # OC
        tr_oracle = np.argmax(ytr, axis=1)
        p_oc = np.array([(tr_oracle==k).mean() for k in range(M)])
        cheap = int(np.argmin(COSTS)); c_nat = float(p_oc @ COSTS)
        if c_nat > BUDGET_EVAL:
            beta = float(np.clip((c_nat-BUDGET_EVAL)/(c_nat-COSTS[cheap]+1e-12), 0, 1))
            e_c = np.zeros(M); e_c[cheap] = 1.0
            p_oc = (1-beta)*p_oc + beta*e_c
        assign_oc = np.random.default_rng(seed).choice(M, size=n_test, p=p_oc)
        avg_q_oc  = float(np.mean(y_test[np.arange(n_test), assign_oc]))
        avg_c_oc  = float(np.mean(COSTS[assign_oc]))
        audc["OC"].append(avg_q_oc)
        acc["OC"].append(float(np.mean(assign_oc == oracle_choice)))

        if seed == seeds[0]:
            curves = {"KNN":(ck,qk),"TFMLP":(ct,qt),"RIDGE":(cr,qr),
                      "LP":(np.array([avg_c_lp]),np.array([avg_q_lp])),
                      "OC":(np.array([avg_c_oc]),np.array([avg_q_oc])),
                      "oracle":(oc,oq)}
            detail = {
                "train_pool": size, "seed": int(seed),
                "route_keys": [f"{m}-{s}" for m,s in ROUTE_KEYS],
                "costs": COSTS.tolist(),
                "records": [{
                    "instruction": test_data[i]["instruction"],
                    "gt_scores": [float(v) for v in y_test[i]],
                    "oracle_choice": int(oracle_choice[i]),
                    "KNN":   {"pred": rs_knn[i].tolist(), "choice": int(np.argmax(rs_knn[i]))},
                    "TFMLP": {"pred": rs_tf[i].tolist(),  "choice": int(np.argmax(rs_tf[i]))},
                    "RIDGE": {"pred": rs_rg[i].tolist(),  "choice": int(np.argmax(rs_rg[i]))},
                    "LP":    {"choice": int(assign_lp[i])},
                    "OC":    {"choice": int(assign_oc[i])},
                } for i in range(n_test)],
            }

        print(f"  seed={seed}  KNN={audc['KNN'][-1]:.4f}  TFMLP={audc['TFMLP'][-1]:.4f}"
              f"  RIDGE={audc['RIDGE'][-1]:.4f}  LP={audc['LP'][-1]:.4f}", flush=True)

    print(f"\n  {'router':22s}  {'@50':>8s}  %oracle   ACC")
    print(f"  {'oracle':22s}  {audc_oracle:8.4f}  100.0%")
    tags = {"KNN":"CATImage-KNN","TFMLP":"CATImage-TFMLP",
            "RIDGE":"Qwen3-8B+Ridge","LP":"LP-train","OC":"OC-train"}
    for name in ["KNN","TFMLP","RIDGE","LP","OC"]:
        mv = np.mean(audc[name]); ma = np.mean(acc[name])
        print(f"  {tags[name]:22s}  {mv:8.4f}  {mv/audc_oracle*100:5.1f}%  {ma*100:5.1f}%")

    # 图
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    sty = {"oracle":("k","--","oracle"),"KNN":("tab:blue","-","KNN"),
           "TFMLP":("tab:green","-","TFMLP"),"RIDGE":("tab:purple","-","Qwen3-8B+Ridge"),
           "LP":("tab:orange","-","LP-train"),"OC":("tab:red","-","OC-train")}
    for key, (c, q) in curves.items():
        col, ls, lab = sty[key]; ax.plot(c, q, ls, color=col, label=lab)
    for i, ((model, cfg), cost) in enumerate(zip(ROUTE_KEYS, COSTS)):
        ax.scatter([cost], [float(np.mean(y_test[:,i]))], marker="x", s=80, zorder=5,
                   label=f"fixed {model}-{cfg}({cost:.0f})")
    ax.axvline(BUDGET_EVAL, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("avg cost"); ax.set_ylabel("avg quality (UniGen)")
    ax.set_title(f"4-tier UniGen  train={size}  budget={BUDGET_EVAL}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    out_fig = os.path.join(PLOT_DIR, f"deferral_4tier_unigen_train{size}.png")
    fig.savefig(out_fig, dpi=110); plt.close()
    print(f"  [plot] {out_fig}")

    pred_path = os.path.join(PRED_DIR, f"preds_4tier_unigen_train{size}.json")
    with open(pred_path, "w", encoding="utf-8") as pf:
        json.dump(detail, pf, ensure_ascii=False, indent=1)
    print(f"  [preds] {pred_path}")

    fixed_avgs = [float(np.mean(y_test[:,i])) for i in range(M)]
    return {
        "size": size, "n_test": n_test, "oracle": audc_oracle,
        "KNN":   {"mean": float(np.mean(audc["KNN"])),   "acc": float(np.mean(acc["KNN"]))},
        "TFMLP": {"mean": float(np.mean(audc["TFMLP"])), "acc": float(np.mean(acc["TFMLP"]))},
        "RIDGE": {"mean": float(np.mean(audc["RIDGE"])), "acc": float(np.mean(acc["RIDGE"]))},
        "LP":    {"mean": float(np.mean(audc["LP"])),    "acc": float(np.mean(acc["LP"]))},
        "OC":    {"mean": float(np.mean(audc["OC"])),    "acc": float(np.mean(acc["OC"]))},
        "fixed": fixed_avgs,
    }


def save_md(all_results, out_path):
    sizes_str = " / ".join(str(r["size"]) for r in all_results)
    lines = [
        f"# UniGen 4-tier router  train {sizes_str}\n",
        "4 route configs: qwen-10 / qwen-50 / flux-10 / flux-50\n",
        "cost: 10 / 50 / 23.1 / 115.5  budget=50  seeds=0/1/2\n",
        "test: UniGen raw 600  scoring: UniGenBench (Gemini testpoint accuracy)\n",
        "",
        "| train | q10 | q50 | f10 | f50 | oracle@50 | Ridge | TFMLP | KNN | LP | OC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        f = r["fixed"]
        def fq(v): return f"{v*100:.2f}"
        def fm(k): return f"{r[k]['mean']*100:.2f} ({r[k]['acc']*100:.1f}%)"
        lines.append(
            f"| {r['size']} "
            f"| {fq(f[0])} | {fq(f[1])} | {fq(f[2])} | {fq(f[3])} "
            f"| {fq(r['oracle'])} "
            f"| {fm('RIDGE')} | {fm('TFMLP')} | {fm('KNN')} | {fm('LP')} | {fm('OC')} |"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[MD] -> {out_path}")


def save_lc_plot(all_results, out_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sizes = [r["size"] for r in all_results]
    methods = ["RIDGE","TFMLP","KNN","LP","OC"]
    colors  = ["tab:purple","tab:green","tab:blue","tab:orange","tab:red"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, col in zip(methods, colors):
        vals = [r[method]["mean"]*100 for r in all_results]
        ls = "-" if method not in ("LP","OC") else "--"
        ax.plot(sizes, vals, "o"+ls, color=col, label=method, lw=1.8, ms=5)
    oracle_val = all_results[0]["oracle"]*100
    ax.axhline(oracle_val, color="k", ls=":", lw=1.2, label=f"oracle ({oracle_val:.2f}%)")
    ax.set_xlabel("train pool size"); ax.set_ylabel("quality@50 (%)")
    ax.set_title("UniGen 4-tier routing LC  budget=50")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(sizes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110); plt.close()
    print(f"[LC plot] -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--gpu",   type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[400,800,1600,2400])
    ap.add_argument("--out_md", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "results_md", "results_catimage_4tier_unigen.md"))
    a = ap.parse_args()

    results = []
    for size in a.sizes:
        r = run_group(size, a.seeds, a.gpu)
        results.append(r)

    save_md(results, a.out_md)
    if len(results) > 1:
        save_lc_plot(results, os.path.join(PLOT_DIR, "lc_quality_at50.png"))
