#!/usr/bin/env python3
"""Resume the repaired CPU experiments; never submit Hope or encode new features."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
from verify_local_results import LEGACY_RESULTS, check_recorded_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("all", "core", "selection", "legacy"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               MPLBACKEND="Agg")
    # Also set the parent before importing NumPy/Torch-backed loaders.
    os.environ.update(env)
    jobs = []

    def add(number, output, extra=(), threads=False):
        script = next(HERE.glob(f"run_exp{number}_*.py"))
        argv = [str(script), *extra]
        if number not in (23, 24):
            argv += ["--device", "cpu"]
        if threads:
            argv += ["--threads", str(args.threads)]
        jobs.append((argv, output))

    if args.suite in ("all", "legacy"):
        jobs.append(([str(HERE / "rerun_legacy_ridge_cpu.py"), "--threads", str(args.threads)], LEGACY_RESULTS))
    if args.suite in ("all", "core"):
        for dataset in ("gm", "am", "unigen"):
            add(23, f"exp23_10action_distill_{dataset}.json", ["--dataset", dataset])
        add(24, "exp24_action_growth.json")
        add(25, "exp25_cross_dataset_generalization.json")
        add(50, "exp50_joint_supervised_router.json", ["--action-space", "a7"])
        add(50, "exp51_joint_supervised_router_flux3.json", ["--action-space", "flux3"])
        add(52, "exp52_flux3_pooled_lobo.json")
        add(53, "exp53_flux3_public_router_data_ablation.json")
        add(54, "exp54_flux3_pooled_router_data_curve.json")
        add(55, "exp55_flux4_pooled_router_data_curve.json")
        for n in (100, 200, 300, 1000):
            add(56, "exp56_flux4_pooled_1k" + (f"_n{n}" if n != 1000 else "") + ".json",
                ["--train-per-source", str(n)])
        for number, output in [(38, "exp38_router_crossbench_generalization.json"),
                               (39, "exp39_pooled_router_lobo.json"),
                               (40, "exp40_nested_router_mix_grid.json"),
                               (42, "exp42_router_factory_sample_efficiency.json"),
                               (43, "exp43_largebench_router_tail.json")]:
            add(number, output, threads=True)
    if args.suite in ("all", "selection"):
        for number, output in [(31, "exp31_action_subset_selection.json"),
                               (35, "exp35_action_subset_crossbench.json"),
                               (36, "exp36_common_a7_crossbench.json"),
                               (37, "exp37_action_admission_crossbench.json")]:
            add(number, output, threads=True)
    if args.plan:
        import shlex
        for argv, output in jobs:
            print(shlex.join([sys.executable, *argv]))
        return
    import experiment_integrity as integrity
    result_dir = Path(integrity.results_dir())
    recorded = check_recorded_results(result_dir, allow_missing=True)
    # Check content identity even when a completed JSON will be skipped.
    import run_exp38_router_crossbench_generalization as E38
    import run_exp56_flux4_pooled_1k as E56
    E38.load_all_data()
    E56.expanded_small("corebench")
    E56.expanded_small("wise")
    for argv, output in jobs:
        outputs = output if isinstance(output, tuple) else (output,)
        if all(name in recorded and (result_dir / name).exists() for name in outputs):
            print(f"[verified complete] {Path(argv[0]).name}", flush=True)
            continue
        subprocess.run([sys.executable, *argv], env=env, check=True)
    if (result_dir / "exp51_joint_supervised_router_flux3.json").exists():
        subprocess.run([sys.executable, str(HERE / "update_repair_reports.py")], env=env, check=True)


if __name__ == "__main__":
    main()
