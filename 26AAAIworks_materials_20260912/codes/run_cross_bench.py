#!/usr/bin/env python3
"""跨 benchmark 泛化实验（RF + score + edit）。

训练池（每模型 m ∈ {qwen,flux}，混合 2000 条）：
  geneval2/corebench/wise/unigen 各 400 条 edit gen + coco 从 raw 随机抽 400（固定 seed）
测试集（各自独立）：
  geneval2/corebench/wise 的 edit raw 全量 + coco 剩余 1600 raw
只跑 RF、score 策略、edit stage。qwen/flux 分开。
指标：DP@50 + %oracle + ACC(与 oracle 档位一致率) + spearman(tts)。

用法：python run_cross_bench.py --model qwen --gpu 0
结果落盘：codes/cross_bench_results/cross_{model}.json
"""
import os, sys, json, random, argparse
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_geneval2_compare as R  # 复用 load_rename / get_emb / DATA / DP 函数
from embed_lr_dp import allocate_dp_scores, allocate_dp, evaluate_allocation

DATA = R.DATA
COCO_SEED = 20260723          # coco 训练/测试分割固定 seed
TRAIN_BENCHES = ["geneval2", "corebench", "wise", "unigen"]  # 各取 400 gen
TEST_BENCHES = ["geneval2", "corebench", "wise", "coco"]     # 各自 raw 测试
N_PER_BENCH = 400
COCO_N_TRAIN = 400
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_bench_results")
os.makedirs(OUT_DIR, exist_ok=True)


def gen_path(bench, model):
    if bench == "coco":
        return f"{DATA}/coco/{model}_edit_raw.json"   # coco 无 gen，用 raw 分
    return f"{DATA}/{bench}/edit/{bench}_{model}_edit_gen.json"


def raw_path(bench, model):
    if bench == "coco":
        return f"{DATA}/coco/{model}_edit_raw.json"
    return f"{DATA}/{bench}/edit/{bench}_{model}_edit_raw.json"


def coco_split(model):
    """coco 2000 raw → (train_idx 400, test_idx 1600)，固定 seed 不重叠。"""
    d = R.load_rename(raw_path("coco", model))
    idx = list(range(len(d)))
    random.Random(COCO_SEED).shuffle(idx)
    return set(idx[:COCO_N_TRAIN]), set(idx[COCO_N_TRAIN:])


def build_train(model, gpu):
    """混合训练池：4 benchmark 各 400 gen + coco 400。返回 (X, y_score3)。"""
    Xs, ys = [], []
    for bench in TRAIN_BENCHES:
        p = gen_path(bench, model)
        d = R.load_rename(p)
        E = R.get_emb(p, d, gpu)
        take = min(N_PER_BENCH, len(d))
        sel = list(range(len(d)))
        random.Random(COCO_SEED).shuffle(sel)
        sel = sel[:take]
        Xs.append(E[sel])
        ys.append(np.array([[d[i]["scores"][k] for k in ("10steps", "40steps", "tts")] for i in sel]))
        print(f"[train] {bench}/{model}: 取 {take} 条", flush=True)
    # coco 训练片
    p = raw_path("coco", model)
    d = R.load_rename(p); E = R.get_emb(p, d, gpu)
    tr_idx, _ = coco_split(model)
    sel = sorted(tr_idx)
    Xs.append(E[sel])
    ys.append(np.array([[d[i]["scores"][k] for k in ("10steps", "40steps", "tts")] for i in sel]))
    print(f"[train] coco/{model}: 取 {len(sel)} 条", flush=True)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def eval_test(bench, model, rf_predict, sca, gpu):
    p = raw_path(bench, model)
    d = R.load_rename(p); E = R.get_emb(p, d, gpu)
    if bench == "coco":
        _, te_idx = coco_split(model)
        sel = sorted(te_idx)
        d = [d[i] for i in sel]; E = E[sel]
    te = [x["scores"] for x in d]
    s10 = [s["10steps"] for s in te]
    yte_s = np.array([[s["10steps"], s["40steps"], s["tts"]] for s in te])
    yte_d = np.array([[s["40steps"] - s["10steps"], s["tts"] - s["40steps"]] for s in te])
    budget = len(d) * 40
    or_d = allocate_dp([{"delta_40_10": x[0], "delta_tts_40": x[1]} for x in yte_d], s10, budget)
    oracle = evaluate_allocation(or_d, te)

    yp = rf_predict(sca.transform(E))
    alloc = allocate_dp_scores(yp, budget)
    dp = evaluate_allocation(alloc, te)
    accm = float(np.mean([a == o for a, o in zip(alloc, or_d)])) * 100
    sp = spearmanr(yp[:, 2], yte_s[:, 2])[0]
    return {
        "n_test": len(d), "oracle": oracle, "all50": float(np.mean([s["40steps"] for s in te])),
        "dp": dp, "pct_oracle": dp / oracle * 100, "acc": accm, "sp_tts": sp,
        "pstd_tstd": float(yp[:, 2].std() / yte_s[:, 2].std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen", "flux"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    Xtr, ytr = build_train(a.model, a.gpu)
    print(f"[train] 混合训练池: X={Xtr.shape} y={ytr.shape}", flush=True)
    sca = StandardScaler().fit(Xtr)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=16, random_state=a.seed)
    rf.fit(sca.transform(Xtr), ytr)

    out = {"model": a.model, "seed": a.seed, "train_size": len(Xtr),
           "train_benches": TRAIN_BENCHES + ["coco"], "results": {}}
    print(f"\n===== cross-bench  model={a.model}  train={len(Xtr)} (RF/score/edit) =====", flush=True)
    print(f"  {'test':10s} {'n':>5s} {'oracle':>8s} {'DP@50':>8s} {'%oracle':>8s} {'ACC':>7s} {'sp_tts':>8s} {'pstd/tstd':>10s}", flush=True)
    for bench in TEST_BENCHES:
        r = eval_test(bench, a.model, rf.predict, sca, a.gpu)
        out["results"][bench] = r
        print(f"  {bench:10s} {r['n_test']:5d} {r['oracle']:8.4f} {r['dp']:8.4f} "
              f"{r['pct_oracle']:7.1f}% {r['acc']:6.1f}% {r['sp_tts']:+8.3f} {r['pstd_tstd']:10.2f}", flush=True)

    op = os.path.join(OUT_DIR, f"cross_{a.model}.json")
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    print(f"\n[saved] {op}", flush=True)


if __name__ == "__main__":
    main()
