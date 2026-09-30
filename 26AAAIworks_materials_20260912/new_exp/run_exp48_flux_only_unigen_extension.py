#!/usr/bin/env python3
"""Flux-only UniGen Router curve on the existing disjoint 3500-prompt pool.

The construction pool is gen4000_edit_v2's 3500 fully evaluated rows.  It is
disjoint from the fixed raw600 test.  Only Flux-9B {10, 50, TTS/edit} actions
are used and the reported metric is absolute mean Router quality.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import run_exp25_cross_dataset_generalization as E25


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(HERE, "results")
OUTPUT = os.path.join(RES, "exp48_flux_only_unigen_extension.json")
OUTPUT_PNG = os.path.join(RES, "exp48_flux_only_unigen_extension.png")
COMBINED_PNG = os.path.join(RES, "exp48_flux_only_benchmark_extension.png")
EXP47_OUTPUT = os.path.join(RES, "exp47_flux_only_smallbench_extension.json")

RESULTS_ROOT = ("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/"
                "hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results")
POOL_PATH = os.path.join(
    RESULTS_ROOT, "unigen_flux", "eval", "gen4000_edit_v2.json",
)
TEST_PATH = os.path.join(
    RESULTS_ROOT, "unigen_flux", "eval", "raw600_edit_v2.json",
)

ACTION_NAMES = ("flux_10steps", "flux_50steps", "flux_tts")
SCORE_KEYS = ("10steps", "50steps", "tts")
COSTS = np.asarray([4.7, 23.1, 44.9], dtype=np.float64)
BUDGET = 10.0
SEEDS = (42, 43, 44, 45, 46)
K_VALUES = (200, 300, 400, 600, 800, 1000, 1200, 1800, 2400, 3000, 3500)
BOOTSTRAP = 5000


def fit_fixed_alpha_heads(train_x, train_y, alphas, device: str):
    """Refit independent dual Ridge heads on all selected rows."""
    import torch

    feature_mean = train_x.mean(0, dtype=np.float64)
    centered = train_x.astype(np.float64) - feature_mean
    feature_scale = np.sqrt(np.mean(centered * centered, axis=0))
    feature_scale[feature_scale < 1e-12] = 1.0
    xtr = (centered / feature_scale).astype(np.float32)
    ridge_x_mean = xtr.mean(0, dtype=np.float64).astype(np.float32)
    xc = xtr - ridge_x_mean
    y_mean = train_y.mean(0, dtype=np.float64).astype(np.float32)
    yc = train_y.astype(np.float32) - y_mean
    xt = torch.from_numpy(np.ascontiguousarray(xc)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(yc)).to(device)
    gram = xt @ xt.T
    identity = torch.eye(len(xc), dtype=gram.dtype, device=device)
    weights = np.empty((train_x.shape[1], train_y.shape[1]), np.float32)
    alpha_array = np.asarray(alphas)
    for alpha in sorted(set(float(value) for value in alpha_array)):
        columns = np.flatnonzero(alpha_array == alpha)
        chol, info = torch.linalg.cholesky_ex(gram + alpha * identity)
        if int(info.max().item()) != 0:
            raise RuntimeError(f"Ridge failed for alpha={alpha}")
        dual = torch.cholesky_solve(yt[:, columns], chol)
        weights[:, columns] = (xt.T @ dual).cpu().numpy()
    intercept = y_mean - ridge_x_mean @ weights
    del xt, yt, gram, identity
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return E25.RidgeHeads(
        feature_mean, feature_scale, weights, intercept.astype(np.float32),
        alpha_array.astype(np.float64),
        np.full(train_y.shape[1], np.nan, dtype=np.float64),
    )


def choose_alpha_and_refit(train_x, train_y, prediction_blocks, device: str):
    n = len(train_x)
    n_alpha = min(50, max(4, n // 5))
    alpha_idx = np.arange(n_alpha, dtype=np.int64)
    ridge_idx = np.arange(n_alpha, n, dtype=np.int64)
    selected = E25.fit_ridge_heads(
        train_x[ridge_idx], train_y[ridge_idx],
        train_x[alpha_idx], train_y[alpha_idx], device=device,
    )
    final = fit_fixed_alpha_heads(train_x, train_y, selected.alphas, device)
    predictions = [
        final.predict(block).astype(np.float32) for block in prediction_blocks
    ]
    return predictions, selected.alphas, selected.val_r2, n_alpha


def read_scores(path: str):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    instructions, scores = [], []
    for row in rows:
        values = []
        for key in SCORE_KEYS:
            value = row["scores"].get(key)
            if isinstance(value, dict):
                value = value.get("avg")
            if value is None:
                raise ValueError(f"{path}: {row.get('case_id')} missing {key}")
            values.append(float(value))
        instructions.append(row["instruction"])
        scores.append(values)
    if len(set(instructions)) != len(instructions):
        raise AssertionError(f"{path}: duplicate instructions")
    return instructions, np.asarray(scores, dtype=np.float32)


def load_data():
    pool_i, pool_y = read_scores(POOL_PATH)
    test_i, test_y = read_scores(TEST_PATH)
    overlap = set(pool_i).intersection(test_i)
    if overlap:
        raise AssertionError(f"construction/test overlap: {len(overlap)}")
    pool_x = np.load(E25.cache_path(POOL_PATH)).astype(np.float32)
    test_x = np.load(E25.cache_path(TEST_PATH)).astype(np.float32)
    if len(pool_x) != len(pool_y) or len(test_x) != len(test_y):
        raise AssertionError(
            f"embedding mismatch: pool {pool_x.shape}/{pool_y.shape}, "
            f"test {test_x.shape}/{test_y.shape}"
        )
    if len(pool_y) != 3500 or len(test_y) != 600:
        raise AssertionError((len(pool_y), len(test_y)))
    return {
        "pool_i": pool_i, "pool_y": pool_y, "pool_x": pool_x,
        "test_i": test_i, "test_y": test_y, "test_x": test_x,
    }


def fit_indices(seed: int, k: int, n: int):
    order = list(range(n))
    random.Random(480048 + seed).shuffle(order)
    return np.asarray(order[:k], dtype=np.int64)


def cache_path(seed: int, k: int):
    return os.path.join(RES, f"exp48_flux3_unigen_seed{seed}_k{k}.npz")


def fit_or_load(data, seed: int, k: int, device: str, force: bool):
    selected = fit_indices(seed, k, len(data["pool_y"]))
    path = cache_path(seed, k)
    if os.path.exists(path) and not force:
        with np.load(path) as cache:
            if np.array_equal(cache["fit_idx"], selected):
                return path
        raise AssertionError(f"{path}: fit_idx mismatch")
    predictions, alphas, val_r2, n_alpha = choose_alpha_and_refit(
        data["pool_x"][selected], data["pool_y"][selected],
        [data["test_x"]], device,
    )
    np.savez_compressed(
        path, fit_idx=selected, test_prediction=predictions[0],
        alphas=np.asarray(alphas), alpha_val_r2=np.asarray(val_r2),
        internal_alpha_size=np.asarray(n_alpha),
    )
    print(f"[saved] {os.path.basename(path)}", flush=True)
    return path


def evaluate(data, seed: int, k: int):
    with np.load(cache_path(seed, k)) as cache:
        prediction = cache["test_prediction"].astype(np.float32)
    router = E25.route_without_target_labels(
        prediction, data["test_y"], COSTS, BUDGET,
    )
    return {
        "seed": int(seed), "quality": float(router.quality),
        "quality_vector": router.per_prompt.astype(np.float64),
        "cost": float(router.cost),
    }


def aggregate(rows, point_index: int, k: int):
    vector = np.stack([row["quality_vector"] for row in rows]).mean(axis=0)
    ci = E25.bootstrap_ci(vector, BOOTSTRAP, 4801000 + point_index)
    return {
        "k": int(k), "router_quality_mean": float(vector.mean()),
        "router_quality_ci95": [float(x) for x in ci],
        "per_seed_quality": [float(row["quality"]) for row in rows],
        "max_cost": float(max(row["cost"] for row in rows)),
    }


def paired_change(rows_by_k, lower: int, upper: int, seed: int):
    lo = np.stack([x["quality_vector"] for x in rows_by_k[lower]]).mean(0)
    hi = np.stack([x["quality_vector"] for x in rows_by_k[upper]]).mean(0)
    delta = hi - lo
    ci = E25.bootstrap_ci(delta, BOOTSTRAP, seed)
    return {"mean": float(delta.mean()), "ci95": [float(x) for x in ci]}


def render(result):
    points = result["points"]
    x = np.asarray([p["k"] for p in points])
    y = np.asarray([p["router_quality_mean"] for p in points])
    lo = np.asarray([p["router_quality_ci95"][0] for p in points])
    hi = np.asarray([p["router_quality_ci95"][1] for p in points])
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(x, y, marker="o", color="#54A24B", linewidth=2, markersize=6)
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none",
                ecolor="#777777", capsize=3, alpha=.8)
    ax.axvline(2400, color="#999999", linestyle="--", linewidth=1)
    ax.set_title("UniGen Flux-only Router")
    ax.set_xlabel("Construction prompts k")
    ax.set_ylabel("Fixed raw600 mean Router quality")
    ax.grid(axis="y", color="#DDDDDD", linewidth=.8)
    ax.annotate(f"{y[-1]:.4f}", xy=(x[-1], y[-1]),
                xytext=(-32, 16), textcoords="offset points")
    fig.text(.995, .005,
             "Flux-9B {10, 50, TTS}; budget=10; fixed disjoint raw600 test; 5 construction seeds.",
             ha="right", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_combined(result):
    with open(EXP47_OUTPUT, encoding="utf-8") as handle:
        small = json.load(handle)["by_dataset"]
    datasets = [
        ("WISE", small["wise"]["points"], "raw400", "#4C78A8"),
        ("CoreBench", small["corebench"]["points"], "raw400", "#F58518"),
        ("UniGen", result["points"], "raw600", "#54A24B"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2))
    for ax, (name, points, test_name, color) in zip(axes, datasets):
        x = np.asarray([p["k"] for p in points])
        y = np.asarray([p["router_quality_mean"] for p in points])
        lo = np.asarray([p["router_quality_ci95"][0] for p in points])
        hi = np.asarray([p["router_quality_ci95"][1] for p in points])
        ax.plot(x, y, marker="o", color=color, linewidth=2, markersize=5.5)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none",
                    ecolor="#888888", capsize=2.5, alpha=.75)
        ax.set_title(name)
        ax.set_xlabel("Construction prompts k")
        ax.set_ylabel(f"Fixed {test_name} mean Router quality")
        ax.grid(axis="y", color="#DDDDDD", linewidth=.8)
        ax.annotate(f"{y[-1]:.4f}", xy=(x[-1], y[-1]),
                    xytext=(-32, 14), textcoords="offset points")
    fig.suptitle("Flux-9B {10, 50, TTS}: absolute Router quality vs data scale")
    fig.text(.995, .005,
             "Budget=10; fixed disjoint raw test; 5 nested construction seeds; curves start at k=200.",
             ha="right", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(COMBINED_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-models", action="store_true")
    args = parser.parse_args()
    os.makedirs(RES, exist_ok=True)
    data = load_data()
    points, rows_by_k = [], {}
    for point_index, k in enumerate(K_VALUES):
        rows = []
        for seed in SEEDS:
            fit_or_load(data, seed, k, args.device, args.force_models)
            rows.append(evaluate(data, seed, k))
        rows_by_k[k] = rows
        point = aggregate(rows, point_index, k)
        points.append(point)
        print(f"[point] k={k} quality={point['router_quality_mean']:.6f}",
              flush=True)
    result = {
        "experiment": 48, "action_names": list(ACTION_NAMES),
        "costs": COSTS.astype(float).tolist(), "budget": BUDGET,
        "construction_path": POOL_PATH, "construction_n": len(data["pool_y"]),
        "fixed_test_path": TEST_PATH, "fixed_test_n": len(data["test_y"]),
        "construction_test_overlap": 0, "seeds": list(SEEDS),
        "bootstrap": BOOTSTRAP, "points": points,
        "changes": {
            "200_to_3500": paired_change(rows_by_k, 200, 3500, 4802001),
            "2400_to_3500": paired_change(rows_by_k, 2400, 3500, 4802002),
            "3000_to_3500": paired_change(rows_by_k, 3000, 3500, 4802003),
        },
    }
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT)
    render(result)
    render_combined(result)
    print(f"[saved] {OUTPUT}", flush=True)
    print(f"[saved] {OUTPUT_PNG}", flush=True)
    print(f"[saved] {COMBINED_PNG}", flush=True)


if __name__ == "__main__":
    main()
