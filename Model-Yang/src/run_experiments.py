"""
SF-MarketGAN: Full experiment pipeline.

Step 1: Single-class validation (3 parallel)
Step 2: Multi-asset FiLM (1 run)
Step 3: Multi-asset FiLM + SF (1 run)

Auto-shutdown on completion.
"""
import subprocess, os, sys, json, time, re, signal
import numpy as np


def get_best(save_dir):
    best = 999
    try:
        for line in open(f"{save_dir}/train.log"):
            m = re.search(r'best=([\d.]+)', line)
            if m:
                best = min(best, float(m.group(1)))
    except:
        pass
    return best


def run_batch(experiments, max_concurrent=3, label=""):
    """Run a batch of experiments with concurrency control."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    procs = {}
    done = set()
    exp_list = list(experiments.items())
    idx = 0

    while len(done) < len(exp_list):
        while len(procs) < max_concurrent and idx < len(exp_list):
            name, cmd = exp_list[idx]
            save_dir = f"results/{name}"
            os.makedirs(save_dir, exist_ok=True)
            log = open(f"{save_dir}/train.log", "w")
            print(f"  [LAUNCH] {name}")
            p = subprocess.Popen(
                cmd.split(), stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            procs[name] = (p, log)
            idx += 1

        for name in list(procs):
            p, log = procs[name]
            if p.poll() is not None:
                log.close()
                best = get_best(f"results/{name}")
                status = "✓" if p.returncode == 0 else f"✗({p.returncode})"
                print(f"  [{status}] {name}: val_frob={best:.4f}")
                done.add(name)
                del procs[name]

        if procs:
            time.sleep(15)


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("  SF-MarketGAN: Full Experiment Pipeline")
    print("=" * 60)

    # ── Step 1: Single-class (3 parallel) ──
    step1 = {
        "single_stocks": (
            "python -u src/train.py --asset_class stocks"
            " --save_dir results/single_stocks --seed 42"
        ),
        "single_bonds": (
            "python -u src/train.py --asset_class bonds"
            " --save_dir results/single_bonds --seed 42"
        ),
        "single_commodities": (
            "python -u src/train.py --asset_class commodities"
            " --save_dir results/single_commodities --seed 42"
        ),
    }
    run_batch(step1, max_concurrent=3, label="Step 1: Single-Class Validation (FiLM)")

    # ── Step 2: Multi-asset FiLM ──
    step2 = {
        "multi_film": (
            "python -u src/train.py --asset_class all"
            " --save_dir results/multi_film --seed 42"
        ),
    }
    run_batch(step2, max_concurrent=1, label="Step 2: Multi-Asset Joint (FiLM)")

    # ── Step 3: Multi-asset FiLM + SF ──
    step3 = {
        "multi_film_sf": (
            "python -u src/train.py --asset_class all --use_sf"
            " --save_dir results/multi_film_sf --seed 42"
        ),
    }
    run_batch(step3, max_concurrent=1, label="Step 3: Full Model (FiLM + SF)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    all_experiments = [
        ("single_stocks", "20 stocks, FiLM"),
        ("single_bonds", "20 bonds, FiLM"),
        ("single_commodities", "20 commodities, FiLM"),
        ("multi_film", "60 assets, FiLM"),
        ("multi_film_sf", "60 assets, FiLM+SF"),
    ]
    print(f"  {'Name':25s} {'Description':30s} {'val_frob':>10s}")
    print(f"  {'-'*67}")
    for name, desc in all_experiments:
        print(f"  {name:25s} {desc:30s} {get_best(f'results/{name}'):>10.4f}")

    with open("results/summary.json", "w") as f:
        json.dump({name: {"desc": desc, "val_frob": get_best(f"results/{name}")}
                   for name, desc in all_experiments}, f, indent=2)

    print(f"\n✓ Pipeline complete. Shutting down...")
    os.kill(-1, signal.SIGKILL)


if __name__ == "__main__":
    main()
