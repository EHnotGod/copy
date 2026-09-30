#!/usr/bin/env python3
"""§25: zero-shot cross-dataset generalization of the A10 Ridge router.

The source domain alone fits the scaler, Ridge heads, and alpha values.  On a
target domain, the routing multiplier/mix is selected using predictions and
costs only; target true scores are read only after the policy is frozen.

GM <-> AM is a metric/label shift over identical prompts.  Transfers between
GenEval2 (GM/AM) and UniGen are the true domain-shift tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import experiment_integrity as integrity


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = integrity.results_dir()
os.makedirs(RES, exist_ok=True)

DATA = os.path.join(ROOT, "data", "final_data")
RESULTS_ROOT = ("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/"
                "hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/Results")
BBASE = os.path.join(RESULTS_ROOT, "geneval2_boogu", "eval")
CACHE = os.path.join(ROOT, "codes", ".cache_embed")
EMB_MODEL = ("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/"
             "multimodal/zhaobo45/CKPTS/Qwen3-Embedding-8B")

DATASETS = ["gm", "am", "unigen"]
ACTION_NAMES = [
    "qwen_10steps", "qwen_50steps", "qwen_tts",
    "flux_10steps", "flux_50steps", "flux_tts",
    "boogu_10steps", "boogu_50steps", "boogu_tts",
    "flux4b_4steps",
]
COSTS = np.asarray([2.0, 10.0, 30.5, 4.7, 23.1, 44.9,
                    4.3, 21.5, 64.8, 0.455], dtype=np.float64)
QWEN50_IDX = ACTION_NAMES.index("qwen_50steps")
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
N_VAL = 200
BUDGET = 10.0
LAMBDAS = np.unique(np.concatenate([
    np.linspace(0.0, 0.05, 2000),
    np.linspace(0.05, 0.3, 500),
    np.linspace(0.3, 3.0, 200),
    np.asarray([10.0, 100.0, 1e6]),
]))


def source_paths(dataset, model, split):
    """Return one/two raw JSON paths for one score source."""
    if dataset in ("gm", "am"):
        if split == "pool":
            if model == "flux4b":
                return (f"{DATA}/geneval2/direct/"
                        f"geneval2_flux4b_direct_gen2400_{dataset}_perseed.json", None)
            if model == "boogu":
                filename = "gen2400_perseed.json" if dataset == "gm" \
                    else "gen2400_am_v2_perseed.json"
                return os.path.join(BBASE, filename), None
            folder = "edit" if dataset == "gm" else "edit_am"
            return (f"{DATA}/geneval2/{folder}/geneval2_{model}_edit_gen_perseed.json",
                    f"{DATA}/geneval2/{folder}/geneval2_{model}_edit_gen2000_perseed.json")
        if model == "flux4b":
            return (f"{DATA}/geneval2/direct/"
                    f"geneval2_flux4b_direct_raw_{dataset}_perseed.json", None)
        if model == "boogu":
            filename = "raw800_perseed.json" if dataset == "gm" \
                else "raw800_am_perseed.json"
            return os.path.join(BBASE, filename), None
        folder = "edit" if dataset == "gm" else "edit_am"
        return f"{DATA}/geneval2/{folder}/geneval2_{model}_edit_raw_perseed.json", None

    if split == "pool":
        if model == "flux4b":
            return f"{RESULTS_ROOT}/unigen_flux4b/eval/gen2400_direct_v2.json", None
        if model == "boogu":
            return f"{RESULTS_ROOT}/unigenbench_boogu/eval/gen2400_tts_edit_v2.json", None
        return f"{RESULTS_ROOT}/unigen_{model}/eval/gen2400_edit_v2.json", None
    if model == "flux4b":
        return f"{RESULTS_ROOT}/unigen_flux4b/eval/raw600_direct_v2.json", None
    if model == "boogu":
        return f"{RESULTS_ROOT}/unigenbench_boogu/eval/raw600_tts_edit_v2.json", None
    return f"{RESULTS_ROOT}/unigen_{model}/eval/raw600_edit_v2.json", None


def read_source(paths, keys):
    records = []
    for path in paths:
        if path is not None:
            with open(path) as f:
                records.extend(json.load(f))
    out = {}
    for record in records:
        values = {}
        for key in keys:
            value = record["scores"].get(key)
            if isinstance(value, dict):
                value = value.get("avg")
            if value is None:
                break
            values[key] = float(value)
        else:
            out[record["instruction"]] = values
    return out


def load_scores(dataset, split):
    keys = {
        "qwen": ["10steps", "50steps", "tts"],
        "flux": ["10steps", "50steps", "tts"],
        "boogu": ["10steps", "50steps", "tts"],
        "flux4b": ["4steps"],
    }
    sources = {
        model: read_source(source_paths(dataset, model, split), model_keys)
        for model, model_keys in keys.items()
    }
    instructions = sorted(set.intersection(*(set(x) for x in sources.values())))
    scores = np.empty((len(instructions), len(ACTION_NAMES)), dtype=np.float32)
    for row, instruction in enumerate(instructions):
        col = 0
        for model in ["qwen", "flux", "boogu"]:
            for key in keys[model]:
                scores[row, col] = sources[model][instruction][key]
                col += 1
        scores[row, col] = sources["flux4b"][instruction]["4steps"]
    expected = 2400 if split == "pool" else (600 if dataset == "unigen" else 800)
    if len(instructions) != expected:
        raise AssertionError(
            f"{dataset}/{split}: got {len(instructions)}, expected {expected}")
    return instructions, scores


def cache_path(raw_path):
    key = hashlib.md5(
        f"{os.path.abspath(raw_path)}||{EMB_MODEL}".encode()).hexdigest()
    return os.path.join(CACHE, f"emb_{key}.npy")


def load_aligned_embeddings(dataset, split, target_instructions):
    return integrity.align_embeddings(
        source_paths(dataset, "qwen", split), target_instructions,
        lambda path: np.load(cache_path(path)),
    )


@dataclass
class DatasetData:
    pool_instructions: list[str]
    pool_scores: np.ndarray
    pool_embeddings: np.ndarray
    test_instructions: list[str]
    test_scores: np.ndarray
    test_embeddings: np.ndarray


def load_dataset(dataset):
    pool_i, pool_y = load_scores(dataset, "pool")
    test_i, test_y = load_scores(dataset, "test")
    original_n = len(pool_i)
    pool_i, pool_y, _ = integrity.exclude_test_prompts(pool_i, pool_y, test_i)
    pool_x = load_aligned_embeddings(dataset, "pool", pool_i)
    test_x = load_aligned_embeddings(dataset, "test", test_i)
    print(f"[loaded] {dataset}: pool={pool_y.shape}/{pool_x.shape} "
          f"test={test_y.shape}/{test_x.shape}", flush=True)
    data = DatasetData(pool_i, pool_y, pool_x, test_i, test_y, test_x)
    integrity.register_dataset(dataset, data)
    if original_n != len(pool_i):
        print(f"[split] {dataset}: excluded {original_n - len(pool_i)} test prompts", flush=True)
    return data


def split_indices(n, seed):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return np.asarray(indices[N_VAL:]), np.asarray(indices[:N_VAL])


def r2_columns(truth, prediction):
    truth64 = truth.astype(np.float64)
    pred64 = prediction.astype(np.float64)
    residual = np.sum((truth64 - pred64) ** 2, axis=0)
    total = np.sum((truth64 - truth64.mean(0)) ** 2, axis=0)
    return 1.0 - residual / np.maximum(total, 1e-20)


@dataclass
class RidgeHeads:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    alphas: np.ndarray
    val_r2: np.ndarray

    def predict(self, features):
        scaled = ((features.astype(np.float64) - self.feature_mean) /
                  self.feature_scale).astype(np.float32)
        return scaled @ self.weights + self.intercept


def fit_ridge_heads(train_x, train_y, val_x, val_y, device="cpu"):
    """Multi-output dual Ridge, matching independent sklearn Ridge heads."""
    import torch

    feature_mean = train_x.mean(0, dtype=np.float64)
    centered = train_x.astype(np.float64) - feature_mean
    feature_scale = np.sqrt(np.mean(centered * centered, axis=0))
    feature_scale[feature_scale < 1e-12] = 1.0
    xtr = (centered / feature_scale).astype(np.float32)
    xva = ((val_x.astype(np.float64) - feature_mean) /
           feature_scale).astype(np.float32)

    # Ridge(fit_intercept=True): center the already standardized source train.
    ridge_x_mean = xtr.mean(0, dtype=np.float64).astype(np.float32)
    xc = xtr - ridge_x_mean
    y_mean = train_y.mean(0, dtype=np.float64).astype(np.float32)
    yc = train_y.astype(np.float32) - y_mean

    xt = torch.from_numpy(np.ascontiguousarray(xc)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(yc)).to(device)
    gram = xt @ xt.T
    identity = torch.eye(len(xc), dtype=gram.dtype, device=device)
    weights_by_alpha, intercepts_by_alpha, r2_by_alpha = [], [], []
    for alpha in ALPHAS:
        chol, info = torch.linalg.cholesky_ex(gram + alpha * identity)
        if int(info.max().item()) != 0:
            raise RuntimeError(f"Ridge Cholesky failed for alpha={alpha}")
        dual = torch.cholesky_solve(yt, chol)
        weights = (xt.T @ dual).cpu().numpy()
        intercept = y_mean - ridge_x_mean @ weights
        val_prediction = xva @ weights + intercept
        weights_by_alpha.append(weights)
        intercepts_by_alpha.append(intercept)
        r2_by_alpha.append(r2_columns(val_y, val_prediction))
    best = np.argmax(np.stack(r2_by_alpha, axis=0), axis=0)
    weights = np.stack([
        weights_by_alpha[int(best[j])][:, j] for j in range(train_y.shape[1])
    ], axis=1)
    intercept = np.asarray([
        intercepts_by_alpha[int(best[j])][j] for j in range(train_y.shape[1])
    ], dtype=np.float32)
    val_r2 = np.asarray([
        r2_by_alpha[int(best[j])][j] for j in range(train_y.shape[1])
    ])
    del xt, yt, gram, identity
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return RidgeHeads(
        feature_mean, feature_scale, weights, intercept,
        np.asarray([ALPHAS[int(x)] for x in best]), val_r2)


@dataclass
class LambdaPolicy:
    cost: float
    predicted_quality: float
    chosen: np.ndarray
    lam: float


@dataclass
class EvaluatedPolicy:
    quality: float
    cost: float
    allocation: np.ndarray
    per_prompt: np.ndarray
    predicted_quality: float | None
    mixture: dict


def lambda_policies(predictions, costs):
    """Policies built from predictions only; target labels are unavailable here."""
    rows = np.arange(len(predictions))
    best_by_cost = {}
    # Batch only the independent arithmetic; retain the exact lambda order,
    # Python rounding and strict tie rule used by the original scalar sweep.
    # This matters for the many subset policies enumerated in Exp31–37.
    for start in range(0, len(LAMBDAS), 64):
        grid = LAMBDAS[start:start + 64]
        chosen_batch = np.argmax(
            predictions[None, :, :] - grid[:, None, None] * costs[None, None, :],
            axis=2,
        )
        mean_cost = np.mean(costs[chosen_batch], axis=1)
        mean_prediction = np.mean(predictions[rows[None, :], chosen_batch], axis=1)
        for lam, chosen, avg_cost, avg_prediction in zip(grid, chosen_batch, mean_cost, mean_prediction):
            cost, predicted_quality = float(avg_cost), float(avg_prediction)
            key = round(cost, 8)
            old = best_by_cost.get(key)
            if old is None or predicted_quality > old.predicted_quality:
                best_by_cost[key] = LambdaPolicy(
                    cost, predicted_quality, chosen.copy(), float(lam))
    return list(best_by_cost.values())


def select_deploy_policy(predictions, costs, budget):
    """Freeze a budget policy using predictions and costs, never true labels."""
    policies = lambda_policies(predictions, costs)
    feasible = [x for x in policies if x.cost <= budget + 1e-10]
    if not feasible:
        raise RuntimeError("no feasible predicted policy")
    low = max(feasible, key=lambda x: (x.predicted_quality, -x.cost))
    selection = (low, None, 0.0)
    best_predicted = low.predicted_quality
    high = [x for x in policies if x.cost > budget + 1e-10]
    for p0 in feasible:
        for p1 in high:
            weight = (budget - p0.cost) / (p1.cost - p0.cost)
            predicted = ((1.0 - weight) * p0.predicted_quality +
                         weight * p1.predicted_quality)
            if predicted > best_predicted + 1e-12:
                best_predicted = predicted
                selection = (p0, p1, float(weight))
    return selection


def evaluate_frozen_policy(selection, truth, costs):
    """Target labels first enter here, after selection is completely frozen."""
    p0, p1, weight = selection
    rows = np.arange(len(truth))
    v0 = truth[rows, p0.chosen].astype(np.float64)
    a0 = np.bincount(p0.chosen, minlength=len(costs)) / len(truth)
    if p1 is None:
        per_prompt, allocation, cost = v0, a0, p0.cost
    else:
        v1 = truth[rows, p1.chosen].astype(np.float64)
        a1 = np.bincount(p1.chosen, minlength=len(costs)) / len(truth)
        per_prompt = (1.0 - weight) * v0 + weight * v1
        allocation = (1.0 - weight) * a0 + weight * a1
        cost = (1.0 - weight) * p0.cost + weight * p1.cost
    return EvaluatedPolicy(
        float(per_prompt.mean()), float(cost), allocation, per_prompt,
        float((1.0 - weight) * p0.predicted_quality +
              weight * (p1.predicted_quality if p1 is not None
                        else p0.predicted_quality)),
        {"lambda_low": p0.lam,
         "lambda_high": None if p1 is None else p1.lam,
         "high_weight": float(weight)},
    )


def route_without_target_labels(predictions, truth, costs, budget):
    selection = select_deploy_policy(predictions, costs, budget)
    return evaluate_frozen_policy(selection, truth, costs)


def candidate_fixed_mixtures(costs, budget):
    for i, cost in enumerate(costs):
        if cost <= budget + 1e-10:
            allocation = np.zeros(len(costs)); allocation[i] = 1.0
            yield allocation
    for i, j in combinations(range(len(costs)), 2):
        ci, cj = float(costs[i]), float(costs[j])
        if ci > cj:
            i, j, ci, cj = j, i, cj, ci
        if ci <= budget < cj and cj > ci:
            weight = (budget - ci) / (cj - ci)
            allocation = np.zeros(len(costs))
            allocation[i], allocation[j] = 1.0 - weight, weight
            yield allocation


def fixed_lp(source_train_scores, target_truth, costs, budget):
    source_mean = source_train_scores.mean(0)
    candidates = list(candidate_fixed_mixtures(costs, budget))
    allocation = max(
        candidates,
        key=lambda x: (float(x @ source_mean), -float(x @ costs)))
    per_prompt = target_truth.astype(np.float64) @ allocation
    return EvaluatedPolicy(
        float(per_prompt.mean()), float(allocation @ costs), allocation,
        per_prompt, None, {"type": "fixed_mixture"})


def oracle_policy(target_truth, costs, budget):
    # Oracle is deliberately label-aware; it is only the system upper bound.
    return route_without_target_labels(target_truth, target_truth, costs, budget)


def bootstrap_ci(values, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, 100):
        size = min(100, n_bootstrap - start)
        indices = rng.integers(0, n, size=(size, n))
        means[start:start + size] = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def policy_dict(policy):
    return {
        "quality": float(policy.quality),
        "cost": float(policy.cost),
        "allocation": [float(x) for x in policy.allocation],
        "predicted_quality": policy.predicted_quality,
        "mixture": policy.mixture,
    }


def run(seeds, budget, n_bootstrap, device):
    datasets = {name: load_dataset(name) for name in DATASETS}
    # Verified invariant: GM and AM are the same prompts with different labels.
    if datasets["gm"].pool_instructions != datasets["am"].pool_instructions:
        raise AssertionError("GM/AM pool prompts differ")
    if datasets["gm"].test_instructions != datasets["am"].test_instructions:
        raise AssertionError("GM/AM test prompts differ")

    oracle = {
        target: oracle_policy(data.test_scores, COSTS, budget)
        for target, data in datasets.items()
    }
    seed_rows = {(source, target): [] for source in DATASETS for target in DATASETS}
    prompt_vectors = {(source, target): [] for source in DATASETS for target in DATASETS}
    model_meta = []
    prediction_audit = []

    for seed in seeds:
        print(f"\n=== split seed {seed} ===", flush=True)
        splits = {
            name: split_indices(len(data.pool_scores), seed)
            for name, data in datasets.items()
        }
        target_lp = {}
        for target, data in datasets.items():
            train_idx, _ = splits[target]
            target_lp[target] = fixed_lp(
                data.pool_scores[train_idx], data.test_scores, COSTS, budget)

        for source, source_data in datasets.items():
            train_idx, val_idx = splits[source]
            print(f"[fit] source={source} seed={seed} device={device}", flush=True)
            model = fit_ridge_heads(
                source_data.pool_embeddings[train_idx],
                source_data.pool_scores[train_idx],
                source_data.pool_embeddings[val_idx],
                source_data.pool_scores[val_idx], device=device)
            model_meta.append({
                "seed": seed,
                "source": source,
                "alphas": [float(x) for x in model.alphas],
                "val_r2": [float(x) for x in model.val_r2],
            })

            for target, target_data in datasets.items():
                prediction = model.predict(target_data.test_embeddings)
                cross = route_without_target_labels(
                    prediction, target_data.test_scores, COSTS, budget)
                source_lp = fixed_lp(
                    source_data.pool_scores[train_idx],
                    target_data.test_scores, COSTS, budget)
                qwen50_vector = target_data.test_scores[:, QWEN50_IDX].astype(np.float64)
                qwen50 = float(qwen50_vector.mean())
                zero_shot = cross.quality - source_lp.quality
                total_gain = cross.quality - qwen50
                row = {
                    "seed": seed,
                    "source": source,
                    "target": target,
                    "qwen50": qwen50,
                    "source_lp": policy_dict(source_lp),
                    "cross_ridge": policy_dict(cross),
                    "target_lp": policy_dict(target_lp[target]),
                    "oracle": policy_dict(oracle[target]),
                    "zero_shot_learned_gain": float(zero_shot),
                    "total_gain_vs_qwen50": float(total_gain),
                    "oracle_gap": float(oracle[target].quality - cross.quality),
                }
                seed_rows[(source, target)].append(row)
                prompt_vectors[(source, target)].append({
                    "cross": cross.per_prompt,
                    "source_lp": source_lp.per_prompt,
                    "qwen50": qwen50_vector,
                })

                if seed == 42 and source == target:
                    cache = os.path.join(
                        RES, f"exp23_10action_distill_preds_{source}.npz")
                    if os.path.exists(cache):
                        with np.load(cache) as old:
                            old_prediction = old["ridge_10"]
                        delta = prediction - old_prediction
                        prediction_audit.append({
                            "dataset": source,
                            "rmse": float(np.sqrt(np.mean(delta ** 2))),
                            "max_abs": float(np.max(np.abs(delta))),
                        })
                print(f"  {source:7s}->{target:7s} "
                      f"cross={cross.quality:.4f} sourceLP={source_lp.quality:.4f} "
                      f"learned={zero_shot:+.4f} cost={cross.cost:.4f}", flush=True)

        # Retention uses the same seed's in-domain learned gain for each target.
        for target in DATASETS:
            in_row = seed_rows[(target, target)][-1]
            denominator = in_row["zero_shot_learned_gain"]
            for source in DATASETS:
                row = seed_rows[(source, target)][-1]
                row["in_domain_learned_gain"] = float(denominator)
                row["transfer_retention"] = (
                    None if denominator <= 1e-8 else
                    float(row["zero_shot_learned_gain"] / denominator))

    aggregate = {}
    scalar_paths = {
        "qwen50": lambda r: r["qwen50"],
        "source_lp": lambda r: r["source_lp"]["quality"],
        "cross_ridge": lambda r: r["cross_ridge"]["quality"],
        "target_lp": lambda r: r["target_lp"]["quality"],
        "oracle": lambda r: r["oracle"]["quality"],
        "zero_shot_learned_gain": lambda r: r["zero_shot_learned_gain"],
        "in_domain_learned_gain": lambda r: r["in_domain_learned_gain"],
        "total_gain_vs_qwen50": lambda r: r["total_gain_vs_qwen50"],
        "oracle_gap": lambda r: r["oracle_gap"],
        "cost": lambda r: r["cross_ridge"]["cost"],
    }
    for source in DATASETS:
        aggregate[source] = {}
        for target in DATASETS:
            rows = seed_rows[(source, target)]
            item = {"source": source, "target": target, "n_seeds": len(rows)}
            for key, getter in scalar_paths.items():
                values = np.asarray([getter(row) for row in rows], dtype=np.float64)
                item[key] = float(values.mean())
                item[f"{key}_seed_std"] = float(values.std(ddof=0))
            denominator = item["in_domain_learned_gain"]
            item["transfer_retention"] = (
                None if denominator <= 1e-8 else
                float(item["zero_shot_learned_gain"] / denominator))
            vectors = prompt_vectors[(source, target)]
            zero_prompt = np.mean(np.stack([
                x["cross"] - x["source_lp"] for x in vectors
            ]), axis=0)
            total_prompt = np.mean(np.stack([
                x["cross"] - x["qwen50"] for x in vectors
            ]), axis=0)
            item["zero_shot_learned_gain_ci95"] = bootstrap_ci(
                zero_prompt, n_bootstrap, 1000 + DATASETS.index(source) * 10 +
                DATASETS.index(target))
            item["total_gain_vs_qwen50_ci95"] = bootstrap_ci(
                total_prompt, n_bootstrap, 2000 + DATASETS.index(source) * 10 +
                DATASETS.index(target))
            item["cross_allocation"] = np.mean(np.stack([
                np.asarray(row["cross_ridge"]["allocation"]) for row in rows
            ]), axis=0).tolist()
            item["per_seed"] = rows
            aggregate[source][target] = item

    result = {
        "experiment": "exp25_cross_dataset_generalization",
        "datasets": DATASETS,
        "action_names": ACTION_NAMES,
        "costs": COSTS.tolist(),
        "budget": float(budget),
        "seeds": seeds,
        "bootstrap_samples": n_bootstrap,
        "device": device,
        "matrix": aggregate,
        "model_meta": model_meta,
        "prediction_audit_vs_exp23": prediction_audit,
        "notes": [
            "GM and AM have identical pool/test prompts and differ only in score aggregation.",
            "All source scaler/Ridge/alpha fitting uses source train/validation only.",
            "Target routing policies are selected from predictions and costs only; target labels enter only after freezing the policy.",
            "Source-LP is the primary zero-shot baseline because it uses the same source supervision as Cross-Ridge.",
            "Oracle is intentionally target-label-aware and is reported only as a system upper bound.",
        ],
    }
    output = os.path.join(RES, "exp25_cross_dataset_generalization.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--budget", type=float, default=BUDGET)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.budget < float(COSTS.min()):
        raise SystemExit(f"budget must be >= minimum action cost {COSTS.min():g}")
    run(seeds, args.budget, args.bootstrap, args.device)


if __name__ == "__main__":
    main()
