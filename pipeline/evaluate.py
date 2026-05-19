# Usage:
#   python -m pipeline.evaluate --agent mappo --model results/models/mappo/best.pt --episodes 5

import os
import argparse
import csv
import torch
import numpy as np

import config as cfg
from env.sumo_env import SumoEnv
from algorithms.mappo import MAPPO, Actor, Critic
from algorithms.attention_mappo import AttentionActor

AGENT_REGISTRY = {
    "mappo": {"actor": Actor, "critic": Critic},
    "attention_mappo": {"actor": AttentionActor,  "critic": Critic},
}


def evaluate(agent_name, model_path, n_episodes, gui):
    classes = AGENT_REGISTRY[agent_name]
    env = SumoEnv(gui=gui)
    agent = MAPPO(env, actor_class=classes["actor"], critic_class=classes["critic"])
    agent.load(model_path)
    agent.actor.eval()

    os.makedirs(os.path.join("results", "data"), exist_ok=True)

    records = []
    best_ep = None
    best_rps = float("-inf")
    best_traj = None
    best_attn = None

    for ep in range(1, n_episodes + 1):
        local_obs, global_obs, _ = env.reset()
        ep_reward, ep_steps = 0.0, 0

        gap_traj = []    # (T, N)
        relv_traj = []   # (T, N)
        spd_traj = []    # (T, N)
        accel_traj = []  # (T, N)
        attn_traj = []   # (T, N, K) — attention MAPPO only

        done = False
        while not done:
            lo_t = torch.tensor(local_obs, dtype=torch.float32)
            with torch.no_grad():
                actions, _ = agent.actor(lo_t)
                if hasattr(agent.actor, "get_attention_weights"):
                    attn_w = agent.actor.get_attention_weights(lo_t).numpy()  # (N, K)
                    attn_traj.append(attn_w)

            next_lo, next_go, rewards, done, info = env.step(actions.numpy().flatten())

            gap_traj.append(info["gaps"].copy())
            relv_traj.append(info["rel_vs"].copy())
            spd_traj.append(info["speeds"].copy())
            accel_traj.append(info["accels"].copy())

            ep_reward += rewards.sum()
            ep_steps += 1
            local_obs, global_obs = next_lo, next_go

        gap_arr = np.array(gap_traj, dtype=np.float32)     # (T, N)
        relv_arr = np.array(relv_traj, dtype=np.float32)   # (T, N)
        spd_arr = np.array(spd_traj, dtype=np.float32)     # (T, N)
        accel_arr = np.array(accel_traj, dtype=np.float32) # (T, N)

        desired = cfg.D0 + cfg.TAU * spd_arr  # (T, N)
        gap_err = gap_arr - desired            # (T, N)

        # Aggregate metrics
        rmse_gap = float(np.sqrt(np.mean(gap_err ** 2)))
        mean_relv = float(np.mean(np.abs(relv_arr)))
        mean_gap = float(gap_arr.mean())
        mean_speed = float(spd_arr.mean())
        pct_large = float((gap_arr > 50).mean() * 100)
        pct_tight = float(((gap_arr >= 5) & (gap_arr <= 30)).mean() * 100)
        survived = 1 if info.get("reason") == "time_up" else 0
        rps = ep_reward / max(ep_steps, 1)

        # Per-agent RMSE (shape N,)
        rmse_per_agent = np.sqrt(np.mean(gap_err ** 2, axis=0))

        # Gi comfort score — mean(1 - jerk^2 / 100), jerk = Δaccel per step
        jerk = np.diff(accel_arr, axis=0)          # (T-1, N)
        gi_comfort = float(np.mean(1 - jerk ** 2 / 100))

        rec = {
            "episode":         ep,
            "reason":          info.get("reason", ""),
            "steps":           ep_steps,
            "ep_reward":       round(ep_reward, 2),
            "reward_per_step": round(rps, 4),
            "mean_gap":        round(mean_gap, 2),
            "rmse_gap":        round(rmse_gap, 2),
            "mean_abs_relv":   round(mean_relv, 4),
            "mean_speed":      round(mean_speed, 3),
            "gi_comfort":      round(gi_comfort, 4),
            "pct_gap_gt50":    round(pct_large, 1),
            "pct_gap_5_30":    round(pct_tight, 1),
            "survived":        survived,
        }
        # Per-agent RMSE columns
        for i, v in enumerate(rmse_per_agent):
            rec[f"rmse_agent{i}"] = round(float(v), 3)

        records.append(rec)

        print(
            f"Ep {ep} | {rec['reason']:12s} | steps={ep_steps} | "
            f"R/step={rps:+.3f} | gap={mean_gap:.1f}m | "
            f"rmse={rmse_gap:.2f} | |Δv|={mean_relv:.3f} | "
            f"spd={mean_speed:.1f} | Gi={gi_comfort:.3f} | survived={survived}"
        )

        if rps > best_rps:
            best_rps = rps
            best_ep = ep
            best_traj = {
                "gaps":    gap_arr,
                "rel_vs":  relv_arr,
                "speeds":  spd_arr,
                "accels":  accel_arr,
                "desired": desired,
            }
            best_attn = np.array(attn_traj) if attn_traj else None

    env.close()

    # Save summary CSV
    csv_path = os.path.join("results", "data", f"{agent_name}_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Save best-episode trajectory CSV
    _save_trajectory(agent_name, best_traj)

    # Save attention weights CSV (attention MAPPO only)
    if best_attn is not None:
        _save_attention(agent_name, best_attn)

    print(f"\nResults saved to {csv_path}")
    _print_summary(agent_name, records)


def _save_trajectory(agent_name, traj):
    path = os.path.join("results", "data", f"{agent_name}_trajectory.csv")
    T, N = traj["gaps"].shape
    with open(path, "w", newline="") as f:
        fieldnames = (
            ["step"]
            + [f"agent{i}_gap"     for i in range(N)]
            + [f"agent{i}_desired" for i in range(N)]
            + [f"agent{i}_relv"    for i in range(N)]
            + [f"agent{i}_speed"   for i in range(N)]
            + [f"agent{i}_accel"   for i in range(N)]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(T):
            row = {"step": t}
            for i in range(N):
                row[f"agent{i}_gap"]     = round(float(traj["gaps"][t, i]),    3)
                row[f"agent{i}_desired"] = round(float(traj["desired"][t, i]), 3)
                row[f"agent{i}_relv"]    = round(float(traj["rel_vs"][t, i]),  3)
                row[f"agent{i}_speed"]   = round(float(traj["speeds"][t, i]),  3)
                row[f"agent{i}_accel"]   = round(float(traj["accels"][t, i]),  3)
            writer.writerow(row)


def _save_attention(agent_name, attn):
    path = os.path.join("results", "data", f"{agent_name}_attention.csv")
    T, N, K = attn.shape
    with open(path, "w", newline="") as f:
        fieldnames = ["step"] + [f"agent{i}_w{k}" for i in range(N) for k in range(K)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(T):
            row = {"step": t}
            for i in range(N):
                for k in range(K):
                    row[f"agent{i}_w{k}"] = round(float(attn[t, i, k]), 4)
            writer.writerow(row)


def _print_summary(agent_name, records):
    n = len(records)
    survived   = sum(r["survived"]        for r in records)
    mean_gap   = np.mean([r["mean_gap"]        for r in records])
    rmse_gap   = np.mean([r["rmse_gap"]         for r in records])
    mean_relv  = np.mean([r["mean_abs_relv"]    for r in records])
    mean_speed = np.mean([r["mean_speed"]       for r in records])
    gi         = np.mean([r["gi_comfort"]       for r in records])
    mean_rps   = np.mean([r["reward_per_step"]  for r in records])

    N = cfg.N_AGENTS
    rmse_agents = [np.mean([r[f"rmse_agent{i}"] for r in records]) for i in range(N)]

    print(f"\n{'─'*55}")
    print(f"  {agent_name} — {n} episodes")
    print(f"  Survival:          {survived}/{n}")
    print(f"  Mean gap:          {mean_gap:.1f} m")
    print(f"  RMSE gap (agg):    {rmse_gap:.2f} m")
    print(f"  RMSE per agent:    {' | '.join(f'{v:.2f}' for v in rmse_agents)}")
    print(f"  Mean |Δv|:         {mean_relv:.3f} m/s")
    print(f"  Mean speed:        {mean_speed:.2f} m/s")
    print(f"  Gi comfort:        {gi:.4f}")
    print(f"  Reward/step:       {mean_rps:.4f}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    type=str, required=True)
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gui",      action="store_true")
    args = parser.parse_args()
    evaluate(args.agent, args.model, args.episodes, args.gui)
