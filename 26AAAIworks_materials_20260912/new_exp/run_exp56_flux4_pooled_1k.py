#!/usr/bin/env python3
"""Flux4 LOBO pooled Router with 1,000 labeled train prompts per source."""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import experiment_integrity as integrity

import run_exp25_cross_dataset_generalization as E25
import run_exp36_common_a7_crossbench as E36
import run_exp38_router_crossbench_generalization as E38


HERE = os.path.dirname(os.path.abspath(__file__))
RES = integrity.results_dir()
OUTPUT = os.path.join(RES, "exp56_flux4_pooled_1k.json")
DATASETS = E38.DATASETS
SEEDS = E38.DEFAULT_SEEDS
ACTION_INDICES = np.asarray([9, 3, 4, 5], dtype=np.int64)
ACTION_NAMES = tuple(E25.ACTION_NAMES[index] for index in ACTION_INDICES)
COSTS = E25.COSTS[ACTION_INDICES]
TRAIN_N = 1000
ALPHA_N = 50
BOOTSTRAP = 5000
SMALL_TEST_N = {"corebench": 400, "wise": 200}


def expanded_small(dataset: str):
    original = E36.load_small_dataset(dataset)
    order = list(range(len(original.test_scores)))
    # Preserve the published Exp47/49 fixed-test partitions.
    partition_seed = 470047 + (1 if dataset == "corebench" else 0)
    random.Random(partition_seed).shuffle(order)
    test_n = SMALL_TEST_N[dataset]
    test_idx = np.asarray(order[:test_n], dtype=np.int64)
    extra_idx = np.asarray(order[test_n:], dtype=np.int64)
    pool_i = original.pool_instructions + [
        original.test_instructions[index] for index in extra_idx
    ]
    test_i = [original.test_instructions[index] for index in test_idx]
    if set(pool_i).intersection(test_i):
        raise AssertionError(f"{dataset}: construction/test overlap")
    data = E25.DatasetData(
        pool_i,
        np.concatenate([
            original.pool_scores,
            original.test_scores[extra_idx],
        ]),
        np.concatenate([
            original.pool_embeddings,
            original.test_embeddings[extra_idx],
        ]),
        test_i,
        original.test_scores[test_idx],
        original.test_embeddings[test_idx],
    )
    print(
        f"[expanded] {dataset}: pool={len(data.pool_scores)} "
        f"test={len(data.test_scores)} overlap=0",
        flush=True,
    )
    integrity.register_dataset(dataset + "_exp56", data)
    return data


def load_all():
    return {
        "gm": E25.load_dataset("gm"),
        "am": E25.load_dataset("am"),
        "unigen": E25.load_dataset("unigen"),
        "corebench": expanded_small("corebench"),
        "wise": expanded_small("wise"),
    }


def split_indices(dataset: str, total: int, seed: int):
    indices = list(range(total))
    random.Random(560000 + DATASETS.index(dataset) * 100 + seed).shuffle(indices)
    alpha = np.asarray(indices[:ALPHA_N], dtype=np.int64)
    train = np.asarray(indices[ALPHA_N:ALPHA_N + TRAIN_N], dtype=np.int64)
    if len(train) != TRAIN_N or len(alpha) != ALPHA_N:
        raise AssertionError((dataset, total, len(train), len(alpha)))
    return train, alpha


def cache_path(target: str, seed: int, sample_size: int = TRAIN_N):
    return os.path.join(
        RES, f"exp56_flux4_pooled_1k_target-{target}_seed{seed}_n{sample_size}.npz",
    )


def fit_or_load(data, target: str, seed: int, device: str = "cpu", sample_size: int = TRAIN_N):
    if not 1 <= sample_size <= TRAIN_N:
        raise ValueError(sample_size)
    path = cache_path(target, seed, sample_size)
    train_x, train_y, alpha_x, alpha_y = [], [], [], []
    payload = {"action_indices": ACTION_INDICES}
    for source in DATASETS:
        if source == target:
            continue
        train, alpha = split_indices(
            source, len(data[source].pool_scores), seed,
        )
        train = train[:sample_size]
        train_x.append(data[source].pool_embeddings[train])
        train_y.append(data[source].pool_scores[train][:, ACTION_INDICES])
        alpha_x.append(data[source].pool_embeddings[alpha])
        alpha_y.append(data[source].pool_scores[alpha][:, ACTION_INDICES])
        payload[f"train_idx_{source}"] = train
        payload[f"alpha_idx_{source}"] = alpha
    if os.path.exists(path):
        with np.load(path) as cache:
            for key_name, expected in payload.items():
                if not np.array_equal(cache[key_name], expected):
                    raise AssertionError((path, key_name))
            return cache["prediction"].astype(np.float32), {
                "cache": os.path.basename(path),
                "alphas": cache["alphas"].astype(float).tolist(),
                "alpha_val_r2": cache["alpha_val_r2"].astype(float).tolist(),
            }, np.concatenate(train_y)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    alpha_x = np.concatenate(alpha_x)
    alpha_y = np.concatenate(alpha_y)
    print(
        f"[fit] target={target} seed={seed} "
        f"train={len(train_y)} alpha={len(alpha_y)}",
        flush=True,
    )
    model = E25.fit_ridge_heads(
        train_x, train_y, alpha_x, alpha_y, device=device,
    )
    prediction = model.predict(
        data[target].test_embeddings,
    ).astype(np.float32)
    payload.update({
        "prediction": prediction,
        "alphas": model.alphas,
        "alpha_val_r2": model.val_r2,
    })
    np.savez_compressed(path, **payload)
    return prediction, {
        "cache": os.path.basename(path),
        "alphas": model.alphas.astype(float).tolist(),
        "alpha_val_r2": model.val_r2.astype(float).tolist(),
    }, train_y


def run(device: str = "cpu", sample_size: int = TRAIN_N):
    data = load_all()
    data_audit = integrity.validate_data(data)
    rows = {target: [] for target in DATASETS}
    metadata = []
    max_cost = 0.0
    for seed in SEEDS:
        for target in DATASETS:
            prediction, meta, source_train_scores = fit_or_load(
                data, target, seed, device, sample_size,
            )
            truth = data[target].test_scores[:, ACTION_INDICES]
            router = E25.route_without_target_labels(
                prediction, truth, COSTS, E25.BUDGET,
            )
            target_train, _ = split_indices(
                target, len(data[target].pool_scores), seed,
            )
            target_lp = E25.fixed_lp(
                data[target].pool_scores[target_train][:, ACTION_INDICES],
                truth, COSTS, E25.BUDGET,
            )
            public_lp = E25.fixed_lp(
                source_train_scores, truth, COSTS, E25.BUDGET,
            )
            rows[target].append({
                "seed": int(seed),
                "router_quality": float(router.quality),
                "target_lp_quality": float(target_lp.quality),
                "public_lp_quality": float(public_lp.quality),
                "router_minus_target_lp": router.per_prompt - target_lp.per_prompt,
                "public_lp_minus_target_lp": public_lp.per_prompt - target_lp.per_prompt,
                "router_allocation": router.allocation.astype(float).tolist(),
                "target_lp_allocation": target_lp.allocation.astype(float).tolist(),
                "public_lp_allocation": public_lp.allocation.astype(float).tolist(),
                "cost": float(router.cost),
            })
            metadata.append({"target": target, "seed": int(seed), **meta})
            max_cost = max(max_cost, router.cost, target_lp.cost, public_lp.cost)
            print(
                f"[eval] {target:9s} seed={seed} "
                f"router={router.quality:.6f} "
                f"targetLP={target_lp.quality:.6f} "
                f"publicLP={public_lp.quality:.6f}",
                flush=True,
            )

    by_target = {}
    for target_index, target in enumerate(DATASETS):
        target_rows = rows[target]
        router_delta = np.stack([
            row["router_minus_target_lp"] for row in target_rows
        ]).mean(0)
        public_delta = np.stack([
            row["public_lp_minus_target_lp"] for row in target_rows
        ]).mean(0)
        by_target[target] = {
            "router_quality": float(np.mean([
                row["router_quality"] for row in target_rows
            ])),
            "target_lp_quality": float(np.mean([
                row["target_lp_quality"] for row in target_rows
            ])),
            "public_lp_quality": float(np.mean([
                row["public_lp_quality"] for row in target_rows
            ])),
            "router_minus_target_lp": float(router_delta.mean()),
            "router_minus_target_lp_ci95": E25.bootstrap_ci(
                router_delta, BOOTSTRAP, 560100 + target_index,
            ),
            "public_lp_minus_target_lp": float(public_delta.mean()),
            "public_lp_minus_target_lp_ci95": E25.bootstrap_ci(
                public_delta, BOOTSTRAP, 560200 + target_index,
            ),
            "router_allocation": np.mean(np.stack([
                row["router_allocation"] for row in target_rows
            ]), axis=0).tolist(),
            "target_lp_allocation": np.mean(np.stack([
                row["target_lp_allocation"] for row in target_rows
            ]), axis=0).tolist(),
            "public_lp_allocation": np.mean(np.stack([
                row["public_lp_allocation"] for row in target_rows
            ]), axis=0).tolist(),
            "per_seed": [{
                key_name: value for key_name, value in row.items()
                if not isinstance(value, np.ndarray)
            } for row in target_rows],
        }
    result = {
        "experiment": "exp56_flux4_pooled_1k",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "train_per_source": sample_size,
        "target_lp_train_n": TRAIN_N,
        "matched_curve_protocol": "fixed test, fixed target LP1000, nested source prefixes, fixed alpha50",
        "alpha_validation_per_source": ALPHA_N,
        "sources_per_target": 4,
        "action_indices": ACTION_INDICES.astype(int).tolist(),
        "action_names": list(ACTION_NAMES),
        "costs": COSTS.astype(float).tolist(),
        "budget": float(E25.BUDGET),
        "test_sizes": {
            target: int(len(data[target].test_scores)) for target in DATASETS
        },
        "construction_sizes": {
            target: int(len(data[target].pool_scores)) for target in DATASETS
        },
        "by_target": by_target,
        "summary": {
            "macro_router_quality": float(np.mean([
                by_target[target]["router_quality"] for target in DATASETS
            ])),
            "macro_target_lp_quality": float(np.mean([
                by_target[target]["target_lp_quality"] for target in DATASETS
            ])),
            "macro_public_lp_quality": float(np.mean([
                by_target[target]["public_lp_quality"] for target in DATASETS
            ])),
            "macro_router_minus_target_lp": float(np.mean([
                by_target[target]["router_minus_target_lp"]
                for target in DATASETS
            ])),
            "positive_targets": int(sum(
                by_target[target]["router_minus_target_lp"] > 0
                for target in DATASETS
            )),
            "significant_positive_targets": int(sum(
                by_target[target]["router_minus_target_lp_ci95"][0] > 0
                for target in DATASETS
            )),
            "significant_negative_targets": int(sum(
                by_target[target]["router_minus_target_lp_ci95"][1] < 0
                for target in DATASETS
            )),
        },
        "model_meta": metadata,
        "numeric_audit": {
            "all_finite": bool(all(
                np.isfinite(value)
                for row in by_target.values()
                for value in (
                    row["router_quality"], row["target_lp_quality"],
                    row["public_lp_quality"], row["router_minus_target_lp"],
                )
            )),
            "max_policy_cost": float(max_cost),
            "budget_respected": bool(max_cost <= E25.BUDGET + 1e-8),
            "target_labels_used_for_pooled_router_fit": False,
            "construction_test_overlap": sum(
                row["construction_test_overlap"] for row in data_audit.values()),
            "by_dataset": data_audit,
        },
    }
    output = OUTPUT if sample_size == TRAIN_N else OUTPUT.replace(".json", f"_n{sample_size}.json")
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output)
    print(output)
    print(json.dumps(result["summary"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-per-source", type=int, choices=(100, 200, 300, 1000), default=1000)
    args = parser.parse_args()
    run(args.device, args.train_per_source)
