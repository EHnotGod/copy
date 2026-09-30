#!/usr/bin/env python3
"""§26 pre-experiment: Qwen3-VL evaluation embedding as router input.

Primary question: at the same mandatory Flux4B-probe cost, does adding a
frozen Qwen3-VL representation improve a Ridge router over prompt-only input?
The Qwen evaluator compute cost is deliberately excluded in this first
information-value screen and must be measured before claiming net deployment
gain.  Flux generation cost is fully included.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import run_exp25_cross_dataset_generalization as E25


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
Q_CACHE = os.path.join(RES, "exp26_qwen_eval_emb_gm_seed0.npz")
OUTPUT = os.path.join(RES, "exp26_probe_image_router_gm.json")
PRED_OUTPUT = os.path.join(RES, "exp26_probe_image_router_preds_gm.npz")
PROBE_IDX = E25.ACTION_NAMES.index("flux4b_4steps")
PROBE_COST = float(E25.COSTS[PROBE_IDX])
BUDGET = 10.0


def load_feature_cache(path, data):
    with np.load(path) as cache:
        pool_i = cache["pool_instructions"].astype(str).tolist()
        test_i = cache["test_instructions"].astype(str).tolist()
        pool_x = cache["pool_embeddings"].astype(np.float32)
        test_x = cache["test_embeddings"].astype(np.float32)
    if pool_i != data.pool_instructions or test_i != data.test_instructions:
        raise AssertionError(f"feature cache instruction mismatch: {path}")
    return pool_x, test_x


def cascade_costs():
    costs = E25.COSTS + PROBE_COST
    costs[PROBE_IDX] = PROBE_COST  # reuse the already generated probe image
    return costs


def top1_accuracy(truth, prediction):
    return float(np.mean(np.argmax(truth, axis=1) == np.argmax(prediction, axis=1)))


def r2_summary(truth, prediction):
    per_action = E25.r2_columns(truth, prediction)
    return {
        "macro": float(per_action.mean()),
        "per_action": {
            name: float(value)
            for name, value in zip(E25.ACTION_NAMES, per_action)
        },
        "top1_action_accuracy": top1_accuracy(truth, prediction),
    }


def route_dict(policy):
    out = E25.policy_dict(policy)
    out["quality"] = float(policy.quality)
    return out


def fit_condition(name, pool_features, test_features, data, train_idx, val_idx,
                  device, seed):
    print(f"[fit] seed={seed} {name}: features={pool_features.shape[1]}",
          flush=True)
    model = E25.fit_ridge_heads(
        pool_features[train_idx], data.pool_scores[train_idx],
        pool_features[val_idx], data.pool_scores[val_idx], device=device)
    prediction = model.predict(test_features)
    val_prediction = model.predict(pool_features[val_idx])
    return {
        "name": name,
        "prediction": prediction,
        "alphas": model.alphas,
        "val": r2_summary(data.pool_scores[val_idx], val_prediction),
        "test": r2_summary(data.test_scores, prediction),
    }


def make_comparisons(primary_name, primary, cost_control, current, lp_probe,
                     bootstrap, seed_base):
    return {
        "information_gain": {
            "value": float(primary.quality - cost_control.quality),
            "ci95": E25.bootstrap_ci(
                primary.per_prompt - cost_control.per_prompt,
                bootstrap, seed_base + 1),
            "definition": (
                f"{primary_name}(cascade cost) - Prompt(cascade cost)"),
        },
        "optimistic_e2e_gain_excluding_qeval_compute": {
            "value": float(primary.quality - current.quality),
            "ci95": E25.bootstrap_ci(
                primary.per_prompt - current.per_prompt,
                bootstrap, seed_base + 2),
            "definition": (
                f"{primary_name}(cascade cost) - Prompt(original cost)"),
        },
        "learned_gain_over_probe_lp": {
            "value": float(primary.quality - lp_probe.quality),
            "ci95": E25.bootstrap_ci(
                primary.per_prompt - lp_probe.per_prompt,
                bootstrap, seed_base + 3),
            "definition": f"{primary_name}(cascade cost) - LP(cascade cost)",
        },
    }


def fit_seed(feature_names, features, data, seed, device):
    train_idx, val_idx = E25.split_indices(len(data.pool_scores), seed)
    fitted = {
        name: fit_condition(
            name, features[name][0], features[name][1], data,
            train_idx, val_idx, device, seed)
        for name in feature_names
    }
    return train_idx, val_idx, fitted


def evaluate_seed(fitted, data, train_idx, bootstrap, seed):
    original_costs = E25.COSTS.copy()
    probe_costs = cascade_costs()
    ytest = data.test_scores
    routes = {
        "prompt_original_cost": E25.route_without_target_labels(
            fitted["prompt"]["prediction"], ytest, original_costs, BUDGET),
        "prompt_probe_cost": E25.route_without_target_labels(
            fitted["prompt"]["prediction"], ytest, probe_costs, BUDGET),
    }
    for name in fitted:
        if name != "prompt":
            routes[f"{name}_probe_cost"] = E25.route_without_target_labels(
                fitted[name]["prediction"], ytest, probe_costs, BUDGET)
    lp_probe = E25.fixed_lp(
        data.pool_scores[train_idx], ytest, probe_costs, BUDGET)
    comparisons = make_comparisons(
        "QEval-only", routes["qeval_only_probe_cost"],
        routes["prompt_probe_cost"], routes["prompt_original_cost"],
        lp_probe, bootstrap, 26000 + seed * 10)
    return routes, lp_probe, comparisons


def ranking_accuracy(truth, prediction, baseline_idx):
    keep = np.arange(truth.shape[1]) != baseline_idx
    truth_better = truth[:, keep] > truth[:, baseline_idx:baseline_idx + 1]
    pred_better = prediction[:, keep] > prediction[:, baseline_idx:baseline_idx + 1]
    return float(np.mean(truth_better == pred_better))


def diagnostic_summary(fitted, routes, data, bootstrap):
    pairwise = {}
    for name, item in fitted.items():
        pairwise[name] = {
            "vs_qwen_50steps": ranking_accuracy(
                data.test_scores, item["prediction"],
                E25.ACTION_NAMES.index("qwen_50steps")),
            "vs_flux4b_4steps": ranking_accuracy(
                data.test_scores, item["prediction"], PROBE_IDX),
        }

    qeval = routes["qeval_only_probe_cost"].per_prompt
    prompt = routes["prompt_probe_cost"].per_prompt
    probe_truth = data.test_scores[:, PROBE_IDX].astype(np.float64)
    edges = np.quantile(probe_truth, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for i in range(4):
        mask = ((probe_truth >= edges[i]) &
                (probe_truth <= edges[i + 1] if i == 3
                 else probe_truth < edges[i + 1]))
        bins.append({
            "quartile": i + 1,
            "count": int(mask.sum()),
            "probe_score_range": [float(edges[i]), float(edges[i + 1])],
            "prompt_probe_quality": float(prompt[mask].mean()),
            "qeval_only_probe_quality": float(qeval[mask].mean()),
            "information_gain": float((qeval[mask] - prompt[mask]).mean()),
        })
    matched_static = {}
    for route_name in ["prompt_probe_cost", "qeval_only_probe_cost",
                       "prompt_qeval_probe_cost"]:
        policy = routes[route_name]
        static_per_prompt = (
            data.test_scores.astype(np.float64) @ policy.allocation)
        delta = policy.per_prompt - static_per_prompt
        matched_static[route_name] = {
            "router_quality": float(policy.quality),
            "same_allocation_static_quality": float(static_per_prompt.mean()),
            "instance_routing_gain": float(delta.mean()),
            "instance_routing_gain_ci95": E25.bootstrap_ci(
                delta, bootstrap, 26800 + len(matched_static)),
        }
    return {
        "pairwise_ranking_accuracy": pairwise,
        "probe_true_quality_quartiles": bins,
        "qeval_routed_win_rate_vs_probe": float(np.mean(qeval > probe_truth)),
        "prompt_routed_win_rate_vs_probe": float(np.mean(prompt > probe_truth)),
        "matched_allocation_static_control": matched_static,
    }


def stability_summary(rows, vectors, bootstrap):
    metric_names = [
        "prompt_test_macro_r2", "qeval_test_macro_r2",
        "prompt_original_quality", "prompt_probe_quality",
        "qeval_only_probe_quality", "information_gain",
        "optimistic_e2e_gain_excluding_qeval_compute",
        "learned_gain_over_probe_lp",
    ]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "seed_std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for i, name in enumerate([
        "information_gain",
        "optimistic_e2e_gain_excluding_qeval_compute",
        "learned_gain_over_probe_lp",
    ]):
        mean_per_prompt = np.stack([item[name] for item in vectors]).mean(0)
        aggregate[name]["mean_over_splits_paired_bootstrap_ci95"] = \
            E25.bootstrap_ci(mean_per_prompt, bootstrap, 26900 + i)
    return aggregate


def run(device="cuda:0", bootstrap=1000, seeds=(42, 43, 44)):
    if not os.path.exists(Q_CACHE):
        raise FileNotFoundError(f"run build_exp26_qwen_eval_cache.py first: {Q_CACHE}")
    data = E25.load_dataset("gm")
    q_pool, q_test = load_feature_cache(Q_CACHE, data)
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one split seed is required")
    main_seed = seeds[0]

    prompt_pool, prompt_test = data.pool_embeddings, data.test_embeddings
    oracle_score_pool = data.pool_scores[:, PROBE_IDX:PROBE_IDX + 1]
    oracle_score_test = data.test_scores[:, PROBE_IDX:PROBE_IDX + 1]
    features = {
        "prompt": (prompt_pool, prompt_test),
        "qeval_only": (q_pool, q_test),
        "prompt_qeval": (
            np.concatenate([prompt_pool, q_pool], axis=1),
            np.concatenate([prompt_test, q_test], axis=1)),
        # Non-deployable diagnostic: the target Flux score is privileged.
        "prompt_true_probe_score": (
            np.concatenate([prompt_pool, oracle_score_pool], axis=1),
            np.concatenate([prompt_test, oracle_score_test], axis=1)),
    }
    train_idx, val_idx, fitted = fit_seed(
        list(features), features, data, main_seed, device)
    original_costs = E25.COSTS.copy()
    probe_costs = cascade_costs()
    ytrain = data.pool_scores[train_idx]
    ytest = data.test_scores

    routes, lp_probe, comparisons = evaluate_seed(
        fitted, data, train_idx, bootstrap, main_seed)
    lp_original = E25.fixed_lp(ytrain, ytest, original_costs, BUDGET)
    oracle_original = E25.oracle_policy(ytest, original_costs, BUDGET)
    oracle_probe = E25.oracle_policy(ytest, probe_costs, BUDGET)

    ablation_comparisons = make_comparisons(
        "Prompt+QEval", routes["prompt_qeval_probe_cost"],
        routes["prompt_probe_cost"], routes["prompt_original_cost"],
        lp_probe, bootstrap, 26100 + main_seed * 10)

    stability_rows, stability_vectors = [], []
    for seed in seeds:
        if seed == main_seed:
            seed_fitted, seed_train_idx = fitted, train_idx
            seed_routes, seed_lp, seed_comparisons = \
                routes, lp_probe, comparisons
        else:
            seed_train_idx, _, seed_fitted = fit_seed(
                ["prompt", "qeval_only"], features, data, seed, device)
            seed_routes, seed_lp, seed_comparisons = evaluate_seed(
                seed_fitted, data, seed_train_idx, bootstrap, seed)
        info = seed_comparisons["information_gain"]["value"]
        e2e = seed_comparisons[
            "optimistic_e2e_gain_excluding_qeval_compute"]["value"]
        learned = seed_comparisons["learned_gain_over_probe_lp"]["value"]
        stability_rows.append({
            "seed": seed,
            "prompt_test_macro_r2": seed_fitted["prompt"]["test"]["macro"],
            "qeval_test_macro_r2": seed_fitted["qeval_only"]["test"]["macro"],
            "prompt_original_quality": seed_routes[
                "prompt_original_cost"].quality,
            "prompt_probe_quality": seed_routes["prompt_probe_cost"].quality,
            "qeval_only_probe_quality": seed_routes[
                "qeval_only_probe_cost"].quality,
            "information_gain": info,
            "optimistic_e2e_gain_excluding_qeval_compute": e2e,
            "learned_gain_over_probe_lp": learned,
            "lp_probe_quality": seed_lp.quality,
            "all_route_costs": {
                name: policy.cost for name, policy in seed_routes.items()
            },
        })
        stability_vectors.append({
            "information_gain": (
                seed_routes["qeval_only_probe_cost"].per_prompt -
                seed_routes["prompt_probe_cost"].per_prompt),
            "optimistic_e2e_gain_excluding_qeval_compute": (
                seed_routes["qeval_only_probe_cost"].per_prompt -
                seed_routes["prompt_original_cost"].per_prompt),
            "learned_gain_over_probe_lp": (
                seed_routes["qeval_only_probe_cost"].per_prompt -
                seed_lp.per_prompt),
        })

    result = {
        "experiment": "exp26_probe_image_router",
        "dataset": "gm",
        "split": {
            "train": 2200, "val": 200, "test": 800,
            "primary_seed": main_seed, "stability_seeds": seeds,
        },
        "budget": BUDGET,
        "probe_action": E25.ACTION_NAMES[PROBE_IDX],
        "probe_generation_cost": PROBE_COST,
        "original_costs": original_costs.tolist(),
        "cascade_costs_excluding_qeval_compute": probe_costs.tolist(),
        "qeval_compute_cost_accounted": False,
        "qeval_feature": (
            "Qwen3-VL backbone final non-padding token under fixed evaluation prompt"),
        "primary_deployable_condition": "qeval_only",
        "conditions": {
            name: {
                "feature_dim": int(features[name][0].shape[1]),
                "alphas": [float(x) for x in item["alphas"]],
                "val": item["val"],
                "test": item["test"],
            }
            for name, item in fitted.items()
        },
        "routes": {name: route_dict(policy) for name, policy in routes.items()},
        "baselines": {
            "lp_original": route_dict(lp_original),
            "lp_probe": route_dict(lp_probe),
            "oracle_original": route_dict(oracle_original),
            "oracle_probe": route_dict(oracle_probe),
        },
        "comparisons": comparisons,
        "ablation_comparisons": {
            "prompt_qeval": ablation_comparisons,
        },
        "diagnostics": diagnostic_summary(fitted, routes, data, bootstrap),
        "split_stability": {
            "per_seed": stability_rows,
            "aggregate": stability_summary(
                stability_rows, stability_vectors, bootstrap),
        },
        "notes": [
            "All route multipliers/mixes are selected from predictions and costs only.",
            "Flux probe generation cost is included; selecting Flux reuses the probe image.",
            "Qwen3-VL evaluation compute is excluded in this information-value pre-experiment, so e2e gain is optimistic.",
            "prompt_true_probe_score is non-deployable privileged-information diagnostics only.",
            "The seed-0 probe image predicts five-seed-average action quality.",
            "QEval-only is primary because the evaluation embedding already conditions on both prompt and image; raw prompt concatenation is an ablation.",
        ],
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        PRED_OUTPUT,
        true_scores=ytest,
        **{f"pred_{name}": item["prediction"] for name, item in fitted.items()},
        **{f"quality_{name}": policy.per_prompt for name, policy in routes.items()},
    )
    print(f"[saved] {OUTPUT}", flush=True)
    print(f"[saved] {PRED_OUTPUT}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    run(args.device, args.bootstrap,
        [int(value) for value in args.seeds.split(",")])


if __name__ == "__main__":
    main()
