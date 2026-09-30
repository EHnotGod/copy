#!/usr/bin/env python3
"""Per-source sample-size curve for one pooled Flux3 Router."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import experiment_integrity as integrity

import run_exp25_cross_dataset_generalization as E25
import run_exp38_router_crossbench_generalization as E38
import run_exp39_pooled_router_lobo as E39
import run_exp52_flux3_pooled_lobo as E52


HERE = os.path.dirname(os.path.abspath(__file__))
RES = integrity.results_dir()
OUTPUT = os.path.join(RES, "exp54_flux3_pooled_router_data_curve.json")
SAMPLE_SIZES = (100, 200, 300)
BOOTSTRAP = 5000


def cache_path(target: str, seed: int, sample_size: int):
    return os.path.join(
        RES,
        f"exp54_flux3_pooled_target-{target}_seed{seed}_n{sample_size}.npz",
    )


def blocks(data, target: str, seed: int, sample_size: int):
    train_x, train_y, alpha_x, alpha_y = [], [], [], []
    indices = {}
    for source in E52.DATASETS:
        if source == target:
            continue
        train, alpha = E39.balanced_indices(
            source, len(data[source].pool_scores), seed,
        )
        train = train[:sample_size]
        train_x.append(data[source].pool_embeddings[train])
        train_y.append(data[source].pool_scores[train][:, E52.INDICES])
        alpha_x.append(data[source].pool_embeddings[alpha])
        alpha_y.append(data[source].pool_scores[alpha][:, E52.INDICES])
        indices[source] = {"train": train, "alpha": alpha}
    return (
        np.concatenate(train_x), np.concatenate(train_y),
        np.concatenate(alpha_x), np.concatenate(alpha_y), indices,
    )


def prediction(data, target: str, seed: int, sample_size: int, device="cpu"):
    if sample_size == 300 and os.path.exists(E52.cache_path(target, seed)):
        path = E52.cache_path(target, seed)
        with np.load(path) as cache:
            return cache["prediction_raw_score"].astype(np.float32), {
                "cache": os.path.basename(path), "reused_exp52": True,
            }
    path = cache_path(target, seed, sample_size)
    train_x, train_y, alpha_x, alpha_y, indices = blocks(
        data, target, seed, sample_size,
    )
    if os.path.exists(path):
        with np.load(path) as cache:
            return cache["prediction"].astype(np.float32), {
                "cache": os.path.basename(path), "reused_exp52": False,
                "alphas": cache["alphas"].astype(float).tolist(),
                "alpha_val_r2": cache["alpha_val_r2"].astype(float).tolist(),
            }
    print(
        f"[fit] target={target} seed={seed} n/source={sample_size} "
        f"train={len(train_y)} alpha={len(alpha_y)}",
        flush=True,
    )
    model = E25.fit_ridge_heads(
        train_x, train_y, alpha_x, alpha_y, device=device,
    )
    pred = model.predict(data[target].test_embeddings).astype(np.float32)
    payload = {
        "prediction": pred,
        "alphas": model.alphas,
        "alpha_val_r2": model.val_r2,
        "sample_size": np.asarray(sample_size),
    }
    for source, split in indices.items():
        payload[f"train_idx_{source}"] = split["train"]
        payload[f"alpha_idx_{source}"] = split["alpha"]
    np.savez_compressed(path, **payload)
    return pred, {
        "cache": os.path.basename(path), "reused_exp52": False,
        "alphas": model.alphas.astype(float).tolist(),
        "alpha_val_r2": model.val_r2.astype(float).tolist(),
    }


def run(device="cpu"):
    data = E38.load_all_data()
    rows = {
        n: {target: [] for target in E52.DATASETS}
        for n in SAMPLE_SIZES
    }
    metadata = []
    max_cost = 0.0
    for sample_size in SAMPLE_SIZES:
        for seed in E52.SEEDS:
            for target in E52.DATASETS:
                pred, meta = prediction(
                    data, target, seed, sample_size,
                )
                truth = data[target].test_scores[:, E52.INDICES]
                policy = E25.route_without_target_labels(
                    pred, truth, E52.COSTS, E25.BUDGET,
                )
                target_train, _ = E39.balanced_indices(
                    target, len(data[target].pool_scores), seed,
                )
                target_lp = E25.fixed_lp(
                    data[target].pool_scores[target_train][:, E52.INDICES],
                    truth, E52.COSTS, E25.BUDGET,
                )
                max_cost = max(max_cost, policy.cost)
                rows[sample_size][target].append({
                    "seed": int(seed),
                    "router_quality": float(policy.quality),
                    "target_lp_quality": float(target_lp.quality),
                    "delta": policy.per_prompt - target_lp.per_prompt,
                    "cost": float(policy.cost),
                    "allocation": policy.allocation.astype(float).tolist(),
                })
                metadata.append({
                    "sample_size": sample_size, "seed": int(seed),
                    "target": target, **meta,
                })
                print(
                    f"[eval] n={sample_size} {target:9s} seed={seed} "
                    f"router={policy.quality:.6f} "
                    f"targetLP={target_lp.quality:.6f}",
                    flush=True,
                )
    by_size = {}
    for sample_size in SAMPLE_SIZES:
        by_target = {}
        for target in E52.DATASETS:
            target_rows = rows[sample_size][target]
            delta = np.stack([row["delta"] for row in target_rows]).mean(0)
            by_target[target] = {
                "router_quality": float(np.mean([
                    row["router_quality"] for row in target_rows
                ])),
                "target_lp_quality": float(np.mean([
                    row["target_lp_quality"] for row in target_rows
                ])),
                "gain_vs_target_lp": float(delta.mean()),
                "gain_vs_target_lp_ci95": E25.bootstrap_ci(
                    delta, BOOTSTRAP,
                    540000 + sample_size + E52.DATASETS.index(target),
                ),
                "per_seed": [{k: v for k, v in row.items() if k != "delta"}
                             for row in target_rows],
            }
        gains = [by_target[target]["gain_vs_target_lp"]
                 for target in E52.DATASETS]
        by_size[str(sample_size)] = {
            "by_target": by_target,
            "macro_router_quality": float(np.mean([
                by_target[target]["router_quality"]
                for target in E52.DATASETS
            ])),
            "macro_target_lp_quality": float(np.mean([
                by_target[target]["target_lp_quality"]
                for target in E52.DATASETS
            ])),
            "macro_gain_vs_target_lp": float(np.mean(gains)),
            "positive_targets": int(sum(value > 0 for value in gains)),
            "significant_positive_targets": int(sum(
                by_target[target]["gain_vs_target_lp_ci95"][0] > 0
                for target in E52.DATASETS
            )),
            "significant_negative_targets": int(sum(
                by_target[target]["gain_vs_target_lp_ci95"][1] < 0
                for target in E52.DATASETS
            )),
        }
    result = {
        "experiment": "exp54_flux3_pooled_router_data_curve",
        "datasets": list(E52.DATASETS),
        "seeds": list(E52.SEEDS),
        "sample_sizes_per_source": list(SAMPLE_SIZES),
        "alpha_validation_per_source": 50,
        "sources_per_target": 4,
        "action_names": list(E52.ACTION_NAMES),
        "costs": E52.COSTS.astype(float).tolist(),
        "budget": float(E25.BUDGET),
        "by_size": by_size,
        "model_meta": metadata,
        "numeric_audit": {
            "all_finite": bool(all(
                np.isfinite(value)
                for row in by_size.values()
                for value in (
                    row["macro_router_quality"],
                    row["macro_target_lp_quality"],
                    row["macro_gain_vs_target_lp"],
                )
            )),
            "max_policy_cost": float(max_cost),
            "budget_respected": bool(max_cost <= E25.BUDGET + 1e-8),
            "target_labels_used_for_pooled_router_fit": False,
            "target_static_lp_train_size": 300,
        },
    }
    temporary = OUTPUT + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, OUTPUT)
    print(OUTPUT)
    for sample_size, row in by_size.items():
        print(
            f"n={sample_size}: quality={row['macro_router_quality']:.6f} "
            f"gain={row['macro_gain_vs_target_lp']:+.6f} "
            f"positive={row['positive_targets']}/5 "
            f"sig+={row['significant_positive_targets']} "
            f"sig-={row['significant_negative_targets']}"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args().device)
