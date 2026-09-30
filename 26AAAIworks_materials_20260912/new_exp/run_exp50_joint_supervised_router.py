#!/usr/bin/env python3
"""One jointly supervised Router for all five known benchmarks.

Unlike Exp39 LOBO, every known benchmark contributes an equal number of
construction and alpha-validation rows to one shared Ridge Router.  The
action space is fixed to the common A7 for every benchmark.  We compare a
plain shared head, the same single head with benchmark-ID features, and five
specialist heads under the same total per-domain label budget.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import experiment_integrity as integrity

import run_exp25_cross_dataset_generalization as E25
import run_exp38_router_crossbench_generalization as E38
import run_exp39_pooled_router_lobo as E39


HERE = os.path.dirname(os.path.abspath(__file__))
RES = integrity.results_dir()
DATASETS = E38.DATASETS
SEEDS = E38.DEFAULT_SEEDS
TRAIN_PER_DOMAIN = 300
ALPHA_PER_DOMAIN = 50
BOOTSTRAP = 5000
ACTION_INDICES = E38.COMMON_INDICES
ACTION_NAMES = E38.ACTION_NAMES
COSTS = E38.COSTS
BASELINE_LOCAL = E38.QWEN50_LOCAL
EXPERIMENT_TAG = "exp50_joint_supervised_router"
OUTPUT = os.path.join(RES, EXPERIMENT_TAG + ".json")


def configure(action_space: str):
    global ACTION_INDICES, ACTION_NAMES, COSTS, BASELINE_LOCAL
    global EXPERIMENT_TAG, OUTPUT
    if action_space == "a7":
        ACTION_INDICES = E38.COMMON_INDICES
        ACTION_NAMES = E38.ACTION_NAMES
        COSTS = E38.COSTS
        BASELINE_LOCAL = E38.QWEN50_LOCAL
        EXPERIMENT_TAG = "exp50_joint_supervised_router"
    elif action_space == "flux3":
        ACTION_INDICES = np.asarray([3, 4, 5], dtype=np.int64)
        ACTION_NAMES = tuple(E25.ACTION_NAMES[index] for index in ACTION_INDICES)
        COSTS = E25.COSTS[ACTION_INDICES]
        BASELINE_LOCAL = 1  # Flux-50 within Flux {10,50,TTS}.
        EXPERIMENT_TAG = "exp51_joint_supervised_router_flux3"
    else:
        raise ValueError(action_space)
    OUTPUT = os.path.join(RES, EXPERIMENT_TAG + ".json")


def add_domain_id(features: np.ndarray, dataset: str):
    onehot = np.zeros((len(features), len(DATASETS)), dtype=np.float32)
    onehot[:, DATASETS.index(dataset)] = 1.0
    return np.concatenate([features, onehot], axis=1)


def split_data(data, seed: int):
    out = {}
    for dataset in DATASETS:
        train, alpha = E39.balanced_indices(
            dataset, len(data[dataset].pool_scores), seed,
        )
        if len(train) != TRAIN_PER_DOMAIN or len(alpha) != ALPHA_PER_DOMAIN:
            raise AssertionError((dataset, len(train), len(alpha)))
        out[dataset] = (train, alpha)
    return out


def cache_path(seed: int):
    return os.path.join(RES, f"{EXPERIMENT_TAG}_predictions_seed{seed}.npz")


def fit_or_load(data, seed: int, device: str, force: bool):
    path = cache_path(seed)
    split = split_data(data, seed)
    if os.path.exists(path) and not force:
        with np.load(path) as cache:
            if not np.array_equal(cache["common_indices"], ACTION_INDICES):
                raise AssertionError(f"{path}: action mismatch")
            predictions = {
                method: {
                    dataset: cache[f"{method}_{dataset}"].astype(np.float32)
                    for dataset in DATASETS
                } for method in ("joint_plain", "joint_domain", "specialist")
            }
            meta = json.loads(str(cache["meta"].item()))
        return predictions, meta, split

    train_x, train_y, alpha_x, alpha_y = [], [], [], []
    domain_train_x, domain_alpha_x = [], []
    for dataset in DATASETS:
        train, alpha = split[dataset]
        item = data[dataset]
        train_x.append(item.pool_embeddings[train])
        alpha_x.append(item.pool_embeddings[alpha])
        domain_train_x.append(add_domain_id(item.pool_embeddings[train], dataset))
        domain_alpha_x.append(add_domain_id(item.pool_embeddings[alpha], dataset))
        train_y.append(item.pool_scores[train][:, ACTION_INDICES])
        alpha_y.append(item.pool_scores[alpha][:, ACTION_INDICES])
    train_y_all = np.concatenate(train_y)
    alpha_y_all = np.concatenate(alpha_y)

    print(f"[fit] seed={seed} joint_plain train={len(train_y_all)} "
          f"alpha={len(alpha_y_all)}", flush=True)
    plain = E25.fit_ridge_heads(
        np.concatenate(train_x), train_y_all,
        np.concatenate(alpha_x), alpha_y_all, device=device,
    )
    print(f"[fit] seed={seed} joint_domain", flush=True)
    domain = E25.fit_ridge_heads(
        np.concatenate(domain_train_x), train_y_all,
        np.concatenate(domain_alpha_x), alpha_y_all, device=device,
    )

    predictions = {"joint_plain": {}, "joint_domain": {}, "specialist": {}}
    specialist_meta = {}
    for dataset in DATASETS:
        item = data[dataset]
        train, alpha = split[dataset]
        predictions["joint_plain"][dataset] = plain.predict(
            item.test_embeddings,
        ).astype(np.float32)
        predictions["joint_domain"][dataset] = domain.predict(
            add_domain_id(item.test_embeddings, dataset),
        ).astype(np.float32)
        print(f"[fit] seed={seed} specialist={dataset}", flush=True)
        specialist = E25.fit_ridge_heads(
            item.pool_embeddings[train],
            item.pool_scores[train][:, ACTION_INDICES],
            item.pool_embeddings[alpha],
            item.pool_scores[alpha][:, ACTION_INDICES],
            device=device,
        )
        predictions["specialist"][dataset] = specialist.predict(
            item.test_embeddings,
        ).astype(np.float32)
        specialist_meta[dataset] = {
            "alphas": specialist.alphas.astype(float).tolist(),
            "alpha_val_r2": specialist.val_r2.astype(float).tolist(),
        }

    meta = {
        "seed": int(seed),
        "joint_plain": {
            "alphas": plain.alphas.astype(float).tolist(),
            "alpha_val_r2": plain.val_r2.astype(float).tolist(),
        },
        "joint_domain": {
            "alphas": domain.alphas.astype(float).tolist(),
            "alpha_val_r2": domain.val_r2.astype(float).tolist(),
        },
        "specialist": specialist_meta,
    }
    payload = {
        "common_indices": ACTION_INDICES,
        "meta": np.asarray(json.dumps(meta)),
    }
    for method in predictions:
        for dataset in DATASETS:
            payload[f"{method}_{dataset}"] = predictions[method][dataset]
    np.savez_compressed(path, **payload)
    print(f"[saved] {path}", flush=True)
    return predictions, meta, split


def aggregate(rows, vectors, bootstrap: int, seed: int):
    scalar_keys = (
        "baseline50_quality", "global_lp_quality", "target_lp_quality",
        "joint_plain_quality", "joint_domain_quality", "specialist_quality",
        "oracle_quality", "joint_plain_cost", "joint_domain_cost",
        "specialist_cost",
    )
    result = {}
    for key in scalar_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = float(values.mean())
        result[f"{key}_seed_std"] = float(values.std(ddof=0))
    comparisons = (
        ("joint_plain_minus_global_lp", "joint_plain", "global_lp"),
        ("joint_plain_minus_target_lp", "joint_plain", "target_lp"),
        ("joint_domain_minus_target_lp", "joint_domain", "target_lp"),
        ("joint_plain_minus_specialist", "joint_plain", "specialist"),
        ("joint_domain_minus_specialist", "joint_domain", "specialist"),
    )
    for offset, (name, lhs, rhs) in enumerate(comparisons):
        vector = np.stack([
            item[lhs] - item[rhs] for item in vectors
        ]).mean(axis=0)
        result[name] = float(vector.mean())
        result[f"{name}_ci95"] = E25.bootstrap_ci(
            vector, bootstrap, seed + offset,
        )
    for method in ("joint_plain", "joint_domain", "specialist"):
        result[f"{method}_allocation"] = np.mean(np.stack([
            np.asarray(row[f"{method}_allocation"], dtype=np.float64)
            for row in rows
        ]), axis=0).tolist()
    result["per_seed"] = rows
    return result


def run(device: str, force: bool):
    data = E38.load_all_data()
    data_audit = integrity.validate_data(data)
    rows = {dataset: [] for dataset in DATASETS}
    vectors = {dataset: [] for dataset in DATASETS}
    meta = []
    max_cost = 0.0
    for seed in SEEDS:
        prediction, model_meta, split = fit_or_load(
            data, seed, device, force,
        )
        meta.append(model_meta)
        pooled_train = np.concatenate([
            data[dataset].pool_scores[split[dataset][0]][:, ACTION_INDICES]
            for dataset in DATASETS
        ])
        for dataset in DATASETS:
            item = data[dataset]
            truth = item.test_scores[:, ACTION_INDICES]
            train, _ = split[dataset]
            baseline50 = truth[:, BASELINE_LOCAL].astype(np.float64)
            global_lp = E25.fixed_lp(
                pooled_train, truth, COSTS, E25.BUDGET,
            )
            target_lp = E25.fixed_lp(
                item.pool_scores[train][:, ACTION_INDICES],
                truth, COSTS, E25.BUDGET,
            )
            policies = {
                method: E25.route_without_target_labels(
                    prediction[method][dataset], truth,
                    COSTS, E25.BUDGET,
                ) for method in ("joint_plain", "joint_domain", "specialist")
            }
            oracle = E25.oracle_policy(truth, COSTS, E25.BUDGET)
            max_cost = max(max_cost, *(p.cost for p in policies.values()))
            row = {
                "seed": int(seed),
                "baseline50_quality": float(baseline50.mean()),
                "global_lp_quality": float(global_lp.quality),
                "target_lp_quality": float(target_lp.quality),
                "oracle_quality": float(oracle.quality),
            }
            for method, policy in policies.items():
                row[f"{method}_quality"] = float(policy.quality)
                row[f"{method}_cost"] = float(policy.cost)
                row[f"{method}_allocation"] = policy.allocation.astype(float).tolist()
            rows[dataset].append(row)
            vectors[dataset].append({
                "joint_plain": policies["joint_plain"].per_prompt,
                "joint_domain": policies["joint_domain"].per_prompt,
                "specialist": policies["specialist"].per_prompt,
                "global_lp": global_lp.per_prompt,
                "target_lp": target_lp.per_prompt,
            })
            print(
                f"[eval] seed={seed} {dataset:9s} "
                f"joint={policies['joint_plain'].quality:.4f} "
                f"domain={policies['joint_domain'].quality:.4f} "
                f"specialist={policies['specialist'].quality:.4f}",
                flush=True,
            )

    by_dataset = {
        dataset: aggregate(
            rows[dataset], vectors[dataset], BOOTSTRAP,
            500000 + DATASETS.index(dataset) * 100,
        ) for dataset in DATASETS
    }
    macro = {}
    for method in ("joint_plain", "joint_domain", "specialist"):
        macro[f"{method}_quality"] = float(np.mean([
            by_dataset[dataset][f"{method}_quality"] for dataset in DATASETS
        ]))
    for name in (
        "joint_plain_minus_global_lp", "joint_plain_minus_target_lp",
        "joint_domain_minus_target_lp", "joint_plain_minus_specialist",
        "joint_domain_minus_specialist",
    ):
        macro[name] = float(np.mean([
            by_dataset[dataset][name] for dataset in DATASETS
        ]))
    result = {
        "experiment": EXPERIMENT_TAG,
        "question": "one fixed-action joint Router for all five known benchmarks",
        "datasets": list(DATASETS),
        "action_names": list(ACTION_NAMES),
        "costs": COSTS.astype(float).tolist(),
        "budget": float(E25.BUDGET),
        "seeds": [int(seed) for seed in SEEDS],
        "train_per_domain": TRAIN_PER_DOMAIN,
        "alpha_val_per_domain": ALPHA_PER_DOMAIN,
        "joint_train_size": TRAIN_PER_DOMAIN * len(DATASETS),
        "joint_alpha_val_size": ALPHA_PER_DOMAIN * len(DATASETS),
        "by_dataset": by_dataset,
        "macro": macro,
        "model_meta": meta,
        "data_audit": data_audit,
        "prompt_domains": {"geneval2": ["gm", "am"], "unigen": ["unigen"],
                           "corebench": ["corebench"], "wise": ["wise"]},
        "numeric_audit": {
            "all_finite": bool(all(
                np.isfinite(value) for value in macro.values()
            )),
            "max_policy_cost": float(max_cost),
            "budget_respected": bool(max_cost <= E25.BUDGET + 1e-8),
            "joint_models_per_seed": 2,
            "target_labels_included_in_joint_training": True,
            "fixed_action_space_for_every_dataset": True,
        },
        "protocol": {
            "joint_plain": "one shared scaler and seven shared Ridge heads",
            "joint_domain": "same single model with five benchmark-ID intercept features",
            "specialist": "one target-only model per benchmark, same 300/50 label budget",
            "primary_metric": "absolute Router quality at budget 10",
        },
    }
    with open(OUTPUT + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(OUTPUT + ".tmp", OUTPUT)
    print(f"[saved] {OUTPUT}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-space", choices=("a7", "flux3"), default="a7")
    args = parser.parse_args()
    configure(args.action_space)
    run(args.device, args.force)


if __name__ == "__main__":
    main()
