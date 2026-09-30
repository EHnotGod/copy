#!/usr/bin/env python3
"""§36: common A7 action-library selection on five benchmarks.

WISE and CoreBench do not have Boogu actions, so the comparable action space is
Qwen/Flux x {10, 50, TTS} plus Flux4B-4step.  GM, AM, and UniGen reuse the
strict validation splits from §31/§35.  CoreBench and WISE use their 400-item
generated pools split into 300 Ridge train / 50 alpha-val / 50 library-val.
All held-out test labels remain untouched until a library is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import experiment_integrity as integrity

import run_exp25_cross_dataset_generalization as E25
import run_exp31_action_subset_selection as E31
import run_exp35_action_subset_crossbench as E35


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = integrity.results_dir()
EXP31 = os.path.join(RES, "exp31_action_subset_selection.json")
EXP35 = os.path.join(RES, "exp35_action_subset_crossbench.json")
OUTPUT = os.path.join(RES, "exp36_common_a7_crossbench.json")
DATASETS = ("gm", "am", "unigen", "corebench", "wise")
SMALL_DATASETS = ("corebench", "wise")
COMMON_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 9], dtype=np.int64)
COMMON_MASK = sum(1 << int(index) for index in COMMON_INDICES)
COMMON_MASKS = [
    mask for mask in range(1, E31.FULL_MASK + 1)
    if mask & ~COMMON_MASK == 0
]
SMALL_N_ALPHA = 50
SMALL_N_LIBRARY = 50
SMALL_STABILITY_SEEDS = tuple(range(40, 50))


def scalar_score(value):
    if not isinstance(value, dict):
        return float(value)
    if "avg" in value:
        return float(value["avg"])
    if "per_seed" in value:
        return float(np.mean(list(value["per_seed"].values())))
    return float(np.mean(list(value.values())))


def small_paths(dataset: str, model: str, split: str):
    if model in ("qwen", "flux"):
        score = os.path.join(
            ROOT, "data", "final_data", dataset, "edit",
            f"{dataset}_{model}_edit_{split}_perseed.json",
        )
        embedding = os.path.join(
            ROOT, "data", "final_data", dataset, "edit",
            f"{dataset}_{model}_edit_{split}.json",
        )
        return score, embedding
    if dataset == "wise":
        score = os.path.join(
            ROOT, "data", "final_data", dataset, "direct",
            f"wise_flux4b_direct_{split}_perseed.json",
        )
    else:
        score = os.path.join(
            ROOT, "data", "final_data", dataset, "direct",
            f"flux4b_direct_{split}.json",
        )
    return score, None


def read_records(path):
    with open(path) as handle:
        return json.load(handle)


def load_small_scores(dataset: str, split: str):
    q_path, _ = small_paths(dataset, "qwen", split)
    f_path, _ = small_paths(dataset, "flux", split)
    b_path, _ = small_paths(dataset, "flux4b", split)
    q = {row["instruction"]: row for row in read_records(q_path)}
    f = {row["instruction"]: row for row in read_records(f_path)}
    b = {row["instruction"]: row for row in read_records(b_path)}
    instructions = sorted(set(q) & set(f) & set(b))
    expected = 400 if split == "gen" else (
        1080 if dataset == "corebench" else 1000
    )
    if len(instructions) != expected:
        raise AssertionError(
            f"{dataset}/{split}: {len(instructions)} != {expected}"
        )
    scores = np.zeros((len(instructions), len(E25.ACTION_NAMES)),
                      dtype=np.float32)
    for row_index, instruction in enumerate(instructions):
        for local, key in enumerate(("10steps", "50steps", "tts")):
            scores[row_index, local] = scalar_score(
                q[instruction]["scores"][key]
            )
            scores[row_index, 3 + local] = scalar_score(
                f[instruction]["scores"][key]
            )
        b_scores = b[instruction]["scores"]
        b_key = "4steps" if "4steps" in b_scores else "steps4"
        scores[row_index, 9] = scalar_score(b_scores[b_key])
    return instructions, scores


def load_small_embeddings(dataset: str, split: str, instructions):
    _, source = small_paths(dataset, "qwen", split)
    records = read_records(source)
    cache = E25.cache_path(source)
    if not os.path.exists(cache):
        raise FileNotFoundError(cache)
    embedding = np.load(cache)
    if len(records) != len(embedding):
        raise AssertionError(f"embedding length mismatch: {source}")
    row_of = {}
    for index, row in enumerate(records):
        row_of.setdefault(row["instruction"], index)
    missing = [value for value in instructions if value not in row_of]
    if missing:
        raise AssertionError(f"{dataset}/{split}: missing embedding")
    rows = np.asarray([row_of[value] for value in instructions])
    out = np.asarray(embedding[rows], dtype=np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError(f"non-finite embedding: {dataset}/{split}")
    return out


def load_small_dataset(dataset: str):
    pool_i, pool_y = load_small_scores(dataset, "gen")
    test_i, test_y = load_small_scores(dataset, "raw")
    pool_x = load_small_embeddings(dataset, "gen", pool_i)
    test_x = load_small_embeddings(dataset, "raw", test_i)
    print(
        f"[loaded] {dataset}: pool={pool_y.shape}/{pool_x.shape} "
        f"test={test_y.shape}/{test_x.shape}", flush=True,
    )
    data = E25.DatasetData(pool_i, pool_y, pool_x, test_i, test_y, test_x)
    integrity.register_dataset(dataset, data)
    return data


def split_small(n: int, seed: int):
    if n != 400:
        raise AssertionError(n)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    indices = np.asarray(indices, dtype=np.int64)
    library = indices[:SMALL_N_LIBRARY]
    alpha = indices[SMALL_N_LIBRARY:SMALL_N_LIBRARY + SMALL_N_ALPHA]
    train = indices[SMALL_N_LIBRARY + SMALL_N_ALPHA:]
    return train, alpha, library


def small_prediction_cache(dataset: str, seed: int):
    return os.path.join(
        RES, f"exp36_common_a7_predictions_{dataset}_seed{seed}.npz",
    )


def fit_or_load_small(data, dataset: str, seed: int, device: str,
                      threads: int, force: bool):
    path = small_prediction_cache(dataset, seed)
    train, alpha, library = split_small(len(data.pool_scores), seed)
    if os.path.exists(path) and not force:
        with np.load(path) as cache:
            for key, expected in [
                ("train_idx", train), ("alpha_val_idx", alpha),
                ("library_val_idx", library),
            ]:
                if not np.array_equal(cache[key], expected):
                    raise AssertionError(f"{path}: {key} mismatch")
            if cache["pool_instructions"].astype(str).tolist() != \
                    data.pool_instructions:
                raise AssertionError(f"{path}: instruction mismatch")
            val = cache["val_predictions"].astype(np.float32)
            test = cache["test_predictions"].astype(np.float32)
            if not np.isfinite(val).all() or not np.isfinite(test).all():
                raise FloatingPointError(path)
            return E31.PredictionSplit(
                train, alpha, library, {dataset: val}, {dataset: test},
                {dataset: cache["alphas"].astype(np.float64)},
                {dataset: cache["alpha_val_r2"].astype(np.float64)},
            )

    import torch
    if device == "cpu":
        torch.set_num_threads(threads)
    print(f"[fit] {dataset} seed={seed} train=300 alpha=50", flush=True)
    model = E25.fit_ridge_heads(
        data.pool_embeddings[train], data.pool_scores[train][:, COMMON_INDICES],
        data.pool_embeddings[alpha], data.pool_scores[alpha][:, COMMON_INDICES],
        device=device,
    )
    local_val = model.predict(data.pool_embeddings[library]).astype(np.float32)
    local_test = model.predict(data.test_embeddings).astype(np.float32)
    val = np.zeros((len(library), len(E25.ACTION_NAMES)), dtype=np.float32)
    test = np.zeros((len(data.test_scores), len(E25.ACTION_NAMES)),
                    dtype=np.float32)
    val[:, COMMON_INDICES] = local_val
    test[:, COMMON_INDICES] = local_test
    np.savez_compressed(
        path, train_idx=train, alpha_val_idx=alpha,
        library_val_idx=library,
        pool_instructions=np.asarray(data.pool_instructions),
        test_instructions=np.asarray(data.test_instructions),
        val_predictions=val, test_predictions=test,
        alphas=model.alphas, alpha_val_r2=model.val_r2,
    )
    print(f"[saved] {path}", flush=True)
    return E31.PredictionSplit(
        train, alpha, library, {dataset: val}, {dataset: test},
        {dataset: model.alphas}, {dataset: model.val_r2},
    )


def enumerate_common(prediction, truth, dataset: str, seed: int):
    rows = {}
    for mask in COMMON_MASKS:
        policy = E31.route_mask(mask, prediction, truth)
        if policy is not None:
            rows[mask] = {
                "quality": float(policy.quality),
                "cost": float(policy.cost),
                "size": int(mask.bit_count()),
            }
    print(
        f"[subsets] {dataset} seed={seed}: {len(rows)}/{len(COMMON_MASKS)}",
        flush=True,
    )
    return rows


def aggregate_rows(rows_by_seed, seeds=E31.SEEDS):
    common = set.intersection(*(set(rows) for rows in rows_by_seed.values()))
    out = {}
    for mask in sorted(common):
        qualities = np.asarray([
            rows_by_seed[seed][mask]["quality"] for seed in seeds
        ])
        costs = np.asarray([
            rows_by_seed[seed][mask]["cost"] for seed in seeds
        ])
        out[mask] = {
            "mask": int(mask),
            "actions": E31.names_for_mask(mask),
            "size": int(mask.bit_count()),
            "validation_quality_mean": float(qualities.mean()),
            "validation_quality_seed_std": float(qualities.std(ddof=0)),
            "validation_cost_mean": float(costs.mean()),
            "per_seed_quality": {
                str(seed): float(rows_by_seed[seed][mask]["quality"])
                for seed in seeds
            },
        }
    return out


def aggregate_crossbench(rows_by_dataset):
    common = set.intersection(*(
        set(rows_by_dataset[name]) for name in DATASETS
    ))
    out = {}
    for mask in sorted(common):
        per_seed = {
            str(seed): float(np.mean([
                rows_by_dataset[name][mask]["per_seed_quality"][str(seed)]
                for name in DATASETS
            ]))
            for seed in E31.SEEDS
        }
        values = np.asarray(list(per_seed.values()))
        out[mask] = {
            "mask": int(mask),
            "actions": E31.names_for_mask(mask),
            "size": int(mask.bit_count()),
            "validation_quality_mean": float(values.mean()),
            "validation_quality_seed_std": float(values.std(ddof=0)),
            "validation_cost_mean": float(np.mean([
                rows_by_dataset[name][mask]["validation_cost_mean"]
                for name in DATASETS
            ])),
            "per_seed_quality": per_seed,
            "per_dataset_quality": {
                name: rows_by_dataset[name][mask]["validation_quality_mean"]
                for name in DATASETS
            },
        }
    return out


def selection_summary(rows):
    best, fixed = E31.choose_subset(rows, E31.TOLERANCE)
    _, one_se, tolerance = E31.choose_one_se(rows)
    return {
        "exact": E31.selection_record(best, best, 0.0),
        "fixed_tolerance": E31.selection_record(
            best, fixed, E31.TOLERANCE,
        ),
        "one_standard_error": E31.selection_record(
            best, one_se, tolerance,
        ),
    }


def one_se_tolerance(rows):
    best = max(
        rows.values(),
        key=lambda row: (row["validation_quality_mean"], -row["size"]),
    )
    values = np.asarray(list(best["per_seed_quality"].values()))
    return best, float(values.std(ddof=1) / math.sqrt(len(values)))


def robust_one_se(rows_by_dataset):
    best, tolerance = {}, {}
    for name in DATASETS:
        best[name], tolerance[name] = one_se_tolerance(rows_by_dataset[name])
    common = set.intersection(*(
        set(rows_by_dataset[name]) for name in DATASETS
    ))
    candidates = []
    for mask in common:
        gaps = {
            name: float(
                best[name]["validation_quality_mean"]
                - rows_by_dataset[name][mask]["validation_quality_mean"]
            )
            for name in DATASETS
        }
        if all(gaps[name] <= tolerance[name] + 1e-12 for name in DATASETS):
            candidates.append((mask, gaps))
    if not candidates:
        return {"feasible": False, "tolerances": tolerance}
    mask, gaps = min(
        candidates,
        key=lambda item: (
            item[0].bit_count(), np.mean(list(item[1].values())),
            max(item[1].values()), item[0],
        ),
    )
    return {
        "feasible": True,
        "mask": int(mask), "actions": E31.names_for_mask(mask),
        "size": int(mask.bit_count()), "validation_gaps": gaps,
        "tolerances": tolerance, "candidate_count": len(candidates),
    }


def diagnostics_for_seed(data, split, prediction, dataset: str):
    truth = data.pool_scores[split.library_val_idx]
    train_truth = data.pool_scores[split.train_idx]
    pred = prediction
    full_router = E31.route_mask(COMMON_MASK, pred, truth)
    full_oracle = E25.oracle_policy(
        truth[:, COMMON_INDICES], E25.COSTS[COMMON_INDICES], E31.BUDGET,
    )
    r2 = E25.r2_columns(
        truth[:, COMMON_INDICES], pred[:, COMMON_INDICES],
    )
    out = {}
    for local, index in enumerate(COMMON_INDICES):
        name = E25.ACTION_NAMES[index]
        drop = np.delete(COMMON_INDICES, local)
        drop_mask = COMMON_MASK & ~(1 << int(index))
        drop_router = E31.route_mask(drop_mask, pred, truth)
        drop_oracle = E25.oracle_policy(
            truth[:, drop], E25.COSTS[drop], E31.BUDGET,
        )
        out[name] = {
            "router_leave_one_out_delta": float(
                full_router.quality - drop_router.quality
            ),
            "oracle_leave_one_out_delta": float(
                full_oracle.quality - drop_oracle.quality
            ),
            "library_val_r2": float(r2[local]),
            "router_allocation": float(full_router.allocation[local]),
        }
    return out


def aggregate_diagnostics(per_seed):
    out = {}
    for name in [E25.ACTION_NAMES[index] for index in COMMON_INDICES]:
        item = {"cost": float(E25.COSTS[E25.ACTION_NAMES.index(name)])}
        for key in [
            "router_leave_one_out_delta", "oracle_leave_one_out_delta",
            "library_val_r2", "router_allocation",
        ]:
            values = np.asarray([
                per_seed[seed][name][key] for seed in E31.SEEDS
            ])
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_seed_std"] = float(values.std(ddof=0))
        out[name] = item
    return out


def small_stability_experiment(data, dataset: str, device: str, threads: int,
                               force: bool, bootstrap: int):
    predictions, rows = {}, {}
    seed_selected_masks = {}
    for seed in SMALL_STABILITY_SEEDS:
        split = fit_or_load_small(
            data, dataset, seed, device, threads, force,
        )
        predictions[seed] = split
        truth = data.pool_scores[split.library_val_idx]
        rows[seed] = enumerate_common(
            split.val_predictions[dataset], truth, dataset, seed,
        )
        seed_selected_masks[seed] = E31.choose_seed_subset(
            rows[seed], E31.TOLERANCE,
        )

    aggregate = aggregate_rows(rows, SMALL_STABILITY_SEEDS)
    selections = selection_summary(aggregate)
    action_frequency = {}
    for index in COMMON_INDICES:
        action = E25.ACTION_NAMES[index]
        count = sum(
            bool(mask & (1 << int(index)))
            for mask in seed_selected_masks.values()
        )
        action_frequency[action] = {
            "count": int(count),
            "fraction": float(count / len(SMALL_STABILITY_SEEDS)),
        }

    per_seed_quality, per_seed_delta, per_seed_cost = {}, {}, {}
    vectors = []
    for seed in SMALL_STABILITY_SEEDS:
        pred = predictions[seed].test_predictions[dataset]
        selected = E31.route_mask(
            seed_selected_masks[seed], pred, data.test_scores,
        )
        full = E31.route_mask(COMMON_MASK, pred, data.test_scores)
        per_seed_quality[str(seed)] = float(selected.quality)
        per_seed_delta[str(seed)] = float(selected.quality - full.quality)
        per_seed_cost[str(seed)] = float(selected.cost)
        vectors.append(selected.per_prompt - full.per_prompt)
    mean_vector = np.stack(vectors).mean(axis=0)
    aggregate_test = {}
    for offset, rule in enumerate([
        "exact", "fixed_tolerance", "one_standard_error",
    ]):
        mask = selections[rule]["selected_subset"]["mask"]
        qualities, costs, rule_vectors = [], [], []
        for seed in SMALL_STABILITY_SEEDS:
            pred = predictions[seed].test_predictions[dataset]
            selected = E31.route_mask(mask, pred, data.test_scores)
            full = E31.route_mask(COMMON_MASK, pred, data.test_scores)
            qualities.append(selected.quality)
            costs.append(selected.cost)
            rule_vectors.append(selected.per_prompt - full.per_prompt)
        rule_vector = np.stack(rule_vectors).mean(axis=0)
        aggregate_test[rule] = {
            "mask": int(mask), "actions": E31.names_for_mask(mask),
            "quality_mean": float(np.mean(qualities)),
            "quality_seed_std": float(np.std(qualities, ddof=0)),
            "cost_mean": float(np.mean(costs)),
            "delta_vs_full_a7": float(rule_vector.mean()),
            "delta_vs_full_a7_ci95": E25.bootstrap_ci(
                rule_vector, bootstrap,
                36810 + offset + (0 if dataset == "corebench" else 10),
            ),
        }
    return {
        "seeds": list(SMALL_STABILITY_SEEDS),
        "aggregate_selection": selections,
        "seed_specific_fixed_tolerance_masks": {
            str(seed): {
                "mask": int(seed_selected_masks[seed]),
                "actions": E31.names_for_mask(seed_selected_masks[seed]),
            }
            for seed in SMALL_STABILITY_SEEDS
        },
        "seed_specific_action_frequency": action_frequency,
        "seed_specific_test_quality": per_seed_quality,
        "seed_specific_test_delta_vs_full_a7": per_seed_delta,
        "seed_specific_test_cost": per_seed_cost,
        "test_quality_mean": float(np.mean(list(per_seed_quality.values()))),
        "test_quality_seed_std": float(np.std(
            list(per_seed_quality.values()), ddof=0,
        )),
        "test_delta_vs_full_a7_mean": float(mean_vector.mean()),
        "test_delta_vs_full_a7_ci95": E25.bootstrap_ci(
            mean_vector, bootstrap, 36800 + (0 if dataset == "corebench" else 1),
        ),
        "aggregate_selected_library_test": aggregate_test,
    }


def evaluate_test(data, predictions, dataset: str, masks, bootstrap: int):
    policies = {name: [] for name in masks}
    per_seed = []
    for seed in E31.SEEDS:
        row = {"seed": seed, "strategies": {}}
        pred = predictions[seed].test_predictions[dataset]
        for strategy, mask in masks.items():
            policy = E31.route_mask(mask, pred, data.test_scores)
            policies[strategy].append(policy)
            row["strategies"][strategy] = {
                "mask": int(mask), "actions": E31.names_for_mask(mask),
                "quality": float(policy.quality), "cost": float(policy.cost),
                "allocation": E31.expanded_allocation(policy, mask),
            }
        per_seed.append(row)
    aggregate = {}
    full = policies["full_a7"]
    q50 = policies["qwen50"]
    for offset, (strategy, runs) in enumerate(policies.items()):
        quality = np.asarray([run.quality for run in runs])
        cost = np.asarray([run.cost for run in runs])
        delta_full = np.stack([
            run.per_prompt - base.per_prompt
            for run, base in zip(runs, full)
        ]).mean(axis=0)
        delta_q50 = np.stack([
            run.per_prompt - base.per_prompt
            for run, base in zip(runs, q50)
        ]).mean(axis=0)
        aggregate[strategy] = {
            "quality_mean": float(quality.mean()),
            "quality_seed_std": float(quality.std(ddof=0)),
            "cost_mean": float(cost.mean()),
            "delta_vs_full_a7": float(delta_full.mean()),
            "delta_vs_full_a7_ci95": E25.bootstrap_ci(
                delta_full, bootstrap, 36000 + offset * 5,
            ),
            "delta_vs_qwen50": float(delta_q50.mean()),
            "delta_vs_qwen50_ci95": E25.bootstrap_ci(
                delta_q50, bootstrap, 36001 + offset * 5,
            ),
        }
    return {"per_seed": per_seed, "aggregate": aggregate}


def run(device: str, threads: int, force_predictions: bool,
        bootstrap: int):
    for path in (EXP31, EXP35):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    with open(EXP31) as handle:
        exp31 = json.load(handle)
    with open(EXP35) as handle:
        exp35 = json.load(handle)

    data = {
        "gm": E25.load_dataset("gm"),
        "am": E25.load_dataset("am"),
        "unigen": E25.load_dataset("unigen"),
        "corebench": load_small_dataset("corebench"),
        "wise": load_small_dataset("wise"),
    }
    gm_am = {name: data[name] for name in ("gm", "am")}
    old_predictions = {
        seed: E31.fit_or_load_predictions(
            gm_am, seed, device, False, threads,
        )
        for seed in E31.SEEDS
    }
    unigen_predictions = {
        seed: E35.fit_or_load_unigen(
            data["unigen"], seed, device, False, threads,
        )
        for seed in E31.SEEDS
    }
    predictions = {
        "gm": old_predictions, "am": old_predictions,
        "unigen": unigen_predictions,
        "corebench": {
            seed: fit_or_load_small(
                data["corebench"], "corebench", seed, device,
                threads, force_predictions,
            )
            for seed in E31.SEEDS
        },
        "wise": {
            seed: fit_or_load_small(
                data["wise"], "wise", seed, device,
                threads, force_predictions,
            )
            for seed in E31.SEEDS
        },
    }

    rows_by_dataset = {
        name: {
            int(row["mask"]): row
            for row in exp31["all_validation_subsets"][name]
            if int(row["mask"]) & ~COMMON_MASK == 0
        }
        for name in ("gm", "am")
    }
    rows_by_dataset["unigen"] = {
        int(row["mask"]): row
        for row in exp35["unigen_all_validation_subsets"]
        if int(row["mask"]) & ~COMMON_MASK == 0
    }
    diagnostics = {}
    for dataset in DATASETS:
        per_seed_diagnostics = {}
        small_rows = {}
        for seed in E31.SEEDS:
            split = predictions[dataset][seed]
            pred = split.val_predictions[dataset]
            truth = data[dataset].pool_scores[split.library_val_idx]
            if dataset in SMALL_DATASETS:
                small_rows[seed] = enumerate_common(
                    pred, truth, dataset, seed,
                )
            per_seed_diagnostics[seed] = diagnostics_for_seed(
                data[dataset], split, pred, dataset,
            )
        if dataset in SMALL_DATASETS:
            rows_by_dataset[dataset] = aggregate_rows(small_rows)
        diagnostics[dataset] = aggregate_diagnostics(per_seed_diagnostics)

    crossbench = aggregate_crossbench(rows_by_dataset)
    selections = {
        name: selection_summary(rows_by_dataset[name])
        for name in DATASETS
    }
    selections["crossbench_mean"] = selection_summary(crossbench)
    robust = robust_one_se(rows_by_dataset)

    cross_mask = selections["crossbench_mean"]["one_standard_error"][
        "selected_subset"
    ]["mask"]
    test = {}
    for dataset in DATASETS:
        masks = {
            "qwen50": E31.QWEN50_MASK,
            "full_a7": COMMON_MASK,
            "dataset_exact": selections[dataset]["exact"][
                "selected_subset"
            ]["mask"],
            "dataset_fixed_tolerance": selections[dataset][
                "fixed_tolerance"
            ]["selected_subset"]["mask"],
            "dataset_one_se": selections[dataset]["one_standard_error"][
                "selected_subset"
            ]["mask"],
            "crossbench_mean_one_se": cross_mask,
        }
        if robust["feasible"]:
            masks["robust_one_se"] = robust["mask"]
        test[dataset] = evaluate_test(
            data[dataset], predictions[dataset], dataset, masks, bootstrap,
        )

    small_stability = {
        dataset: small_stability_experiment(
            data[dataset], dataset, device, threads,
            force_predictions, bootstrap,
        )
        for dataset in SMALL_DATASETS
    }

    action_summary = {}
    one_se_actions = {
        dataset: set(selections[dataset]["one_standard_error"][
            "selected_subset"
        ]["actions"])
        for dataset in DATASETS
    }
    for action in [E25.ACTION_NAMES[index] for index in COMMON_INDICES]:
        router = {
            dataset: diagnostics[dataset][action][
                "router_leave_one_out_delta_mean"
            ]
            for dataset in DATASETS
        }
        action_summary[action] = {
            "cost": float(E25.COSTS[E25.ACTION_NAMES.index(action)]),
            "one_se_presence": {
                dataset: action in one_se_actions[dataset]
                for dataset in DATASETS
            },
            "one_se_presence_count": int(sum(
                action in one_se_actions[dataset] for dataset in DATASETS
            )),
            "router_loo": router,
            "router_loo_positive_count": int(sum(
                value > 0 for value in router.values()
            )),
            "router_loo_mean": float(np.mean(list(router.values()))),
            "oracle_loo": {
                dataset: diagnostics[dataset][action][
                    "oracle_leave_one_out_delta_mean"
                ]
                for dataset in DATASETS
            },
            "r2": {
                dataset: diagnostics[dataset][action]["library_val_r2_mean"]
                for dataset in DATASETS
            },
            "allocation": {
                dataset: diagnostics[dataset][action][
                    "router_allocation_mean"
                ]
                for dataset in DATASETS
            },
        }

    result = {
        "experiment": "exp36_common_a7_crossbench",
        "datasets": list(DATASETS),
        "common_actions": [
            E25.ACTION_NAMES[index] for index in COMMON_INDICES
        ],
        "common_action_mask": int(COMMON_MASK),
        "costs": {
            E25.ACTION_NAMES[index]: float(E25.COSTS[index])
            for index in COMMON_INDICES
        },
        "budget": E31.BUDGET,
        "seeds": list(E31.SEEDS),
        "split": {
            "gm_am_unigen": "GM/AM2000+200+200; UniGen1600+200+200 after test exclusion",
            "corebench_wise": "300 Ridge / 50 alpha / 50 library",
            "test": {
                "gm": 800, "am": 800, "unigen": 600,
                "corebench": 1080, "wise": 1000,
            },
            "test_labels_used_for_selection": False,
        },
        "selections": selections,
        "robust_one_se": robust,
        "test": test,
        "small_pool_10split_stability": small_stability,
        "action_diagnostics": diagnostics,
        "crossbench_action_summary": action_summary,
        "validation_subsets": {
            dataset: [rows_by_dataset[dataset][mask]
                      for mask in sorted(rows_by_dataset[dataset])]
            for dataset in DATASETS
        },
        "crossbench_mean_validation_subsets": [
            crossbench[mask] for mask in sorted(crossbench)
        ],
        "notes": [
            "A7 is the largest action space with complete outputs on all five benchmarks; WISE/CoreBench have no Boogu actions.",
            "CoreBench/WISE have only 400 generated training prompts, so their 50-item library validation estimates are less stable than the 200-item estimates on the other datasets.",
            "CoreBench/WISE additionally use seeds 40-49 to audit small-pool subset-selection stability.",
            "The Qwen3-Embedding-8B caches and the same Ridge/budget router are used throughout.",
            "All library selection is validation-only; held-out test labels enter after freezing.",
        ],
    }
    with open(OUTPUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[saved] {OUTPUT}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run(args.device, args.threads, args.force_predictions, args.bootstrap)


if __name__ == "__main__":
    main()
