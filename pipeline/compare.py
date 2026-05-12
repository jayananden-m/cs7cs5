# Usage:
#   python -m pipeline.compare --agents mappo attention_mappo

import os
import argparse
import csv
import numpy as np


def load_results(agent_name):
    path = os.path.join("results", "data", f"{agent_name}_eval.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No eval CSV for '{agent_name}' at {path}. Run pipeline/evaluate.py first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def summarise(records):
    n = len(records)
    return {
        "n_episodes": n,
        "survival": f"{sum(int(r['survived']) for r in records)}/{n}",
        "mean_gap": round(np.mean([float(r["mean_gap"]) for r in records]), 1),
        "gap_gt50_pct": round(np.mean([float(r["pct_gap_gt50"]) for r in records]), 1),
        "gap_5_30_pct": round(np.mean([float(r["pct_gap_5_30"]) for r in records]), 1),
        "reward_per_step": round(np.mean([float(r["reward_per_step"]) for r in records]), 4),
    }


def compare(agent_names: list[str]):
    summaries = {}
    for name in agent_names:
        records = load_results(name)
        summaries[name] = summarise(records)

    # Print aligned table
    cols = ["survival", "mean_gap", "gap_gt50_pct", "gap_5_30_pct", "reward_per_step"]
    labels = ["Survival", "Mean Gap (m)", "Gap>50m (%)", "Gap 5-30m (%)", "Reward/step"]
    col_w = max(len(n) for n in agent_names) + 2

    header = f"{'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in agent_names)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    for col, label in zip(cols, labels):
        row = f"{label:<20}"
        for name in agent_names:
            val = str(summaries[name][col])
            row += f"{val:>{col_w}}"
        print(row)

    print("─" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", required=True)
    args = parser.parse_args()
    compare(args.agents)
