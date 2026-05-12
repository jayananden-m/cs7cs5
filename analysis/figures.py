# Usage:
#   python -m analysis.figures
#
# Input:  results/data/*_eval.csv  results/data/*_trajectory.csv
#         results/data/*_attention.csv  results/logs/*/
#
# Output: results/figures/

import os
import re
import csv
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from env.leader_profile import SmoothBrakeLeader
import config as cfg


FIGURES_DIR = os.path.join("results", "figures")
DATA_DIR = os.path.join("results", "data")
LOGS_DIR = os.path.join("results", "logs")

AGENT_COLORS = {
    "mappo": "#636EFA",
    "attention_mappo": "#EF553B",
}

AGENT_LABELS = {
    "mappo": "MAPPO",
    "attention_mappo": "Attention MAPPO",
}

NEIGHBOR_COLORS = ["#636EFA", "#EF553B", "#00CC96"]
NEIGHBOR_LABELS = ["Predecessor (k=1)", "k=2 ahead", "k=3 ahead"]
AGENT_LINE_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]


# Data loaders
def load_eval_csv(agent_name) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{agent_name}_eval.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def load_trajectory_csv(agent_name):
    # Returns (T, N) arrays for gaps, desired, rel_vs
    path = os.path.join(DATA_DIR, f"{agent_name}_trajectory.csv")
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, None

    n_agents = sum(1 for k in rows[0] if k.startswith("agent") and k.endswith("_gap"))
    T = len(rows)
    gaps = np.zeros((T, n_agents), dtype=np.float32)
    desired = np.zeros((T, n_agents), dtype=np.float32)
    relvs = np.zeros((T, n_agents), dtype=np.float32)
    for t, row in enumerate(rows):
        for i in range(n_agents):
            gaps[t, i] = float(row[f"agent{i}_gap"])
            desired[t, i] = float(row[f"agent{i}_desired"])
            relvs[t, i] = float(row[f"agent{i}_relv"])
    return gaps, desired, relvs


def load_attention_csv(agent_name):
    # Returns (T, N, K) array of attention weights
    path = os.path.join(DATA_DIR, f"{agent_name}_attention.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n_agents = sum(1 for k in rows[0] if k.startswith("agent") and "_w0" in k)
    n_weights = sum(1 for k in rows[0] if k.startswith("agent0_w"))
    T = len(rows)
    attn = np.zeros((T, n_agents, n_weights), dtype=np.float32)
    for t, row in enumerate(rows):
        for i in range(n_agents):
            for k in range(n_weights):
                attn[t, i, k] = float(row[f"agent{i}_w{k}"])
    return attn


def parse_training_log(agent_name):
    # Parse training log (episodes, rewards_per_step, critic_losses)
    log_dir = os.path.join(LOGS_DIR, agent_name)
    if not os.path.isdir(log_dir):
        return [], [], []
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".log")], reverse=True)
    if not log_files:
        return [], [], []

    log_path = os.path.join(log_dir, log_files[0])
    pattern = re.compile(r"Ep\s+(\d+).*?R=\s*([-\d.]+).*?steps=\s*(\d+).*?c_loss=([\d.]+)")
    episodes, rps_list, closs_list = [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep, r, steps, c = int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))
                episodes.append(ep)
                rps_list.append(r / max(steps, 1))
                closs_list.append(c)
    return episodes, rps_list, closs_list


def rolling_mean(values: list, window = 20) -> list:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


# Plots
def fig1_reward_curve(agents):
    # Per-step reward training curve (20-ep rolling avg)
    fig = go.Figure()
    for name in agents:
        eps, rps, _ = parse_training_log(name)
        if not eps:
            print(f"  [skip] no training log for {name}")
            continue
        fig.add_trace(go.Scatter(
            x=eps, y=rolling_mean(rps),
            name=AGENT_LABELS.get(name, name),
            line=dict(color=AGENT_COLORS.get(name, "#888"), width=2),
        ))
    fig.update_layout(
        title="Training - Reward per Step (20-ep rolling avg)",
        xaxis_title="Episode",
        yaxis_title="Reward / step",
        template="plotly_white",
        height=450, width=860,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=160),
    )
    _save(fig, "fig1_reward_curve.png")


def fig2_critic_loss(agents):
    # Critic loss training curve (20-ep rolling avg)
    fig = go.Figure()
    for name in agents:
        eps, _, closs = parse_training_log(name)
        if not eps:
            print(f"  [skip] no training log for {name}")
            continue
        fig.add_trace(go.Scatter(
            x=eps, y=rolling_mean(closs),
            name=AGENT_LABELS.get(name, name),
            line=dict(color=AGENT_COLORS.get(name, "#888"), width=2),
        ))
    fig.update_layout(
        title="Training - Critic Loss (20-ep rolling avg)",
        xaxis_title="Episode",
        yaxis_title="Critic loss",
        template="plotly_white",
        height=450, width=860,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=160),
    )
    _save(fig, "fig2_critic_loss.png")


def fig3_gap_trajectory(agents):
    # Best-episode gap trajectory for each agent type
    # Each subplot shows per-vehicle gap and desired gap line.

    valid = [(name, *load_trajectory_csv(name)) for name in agents]
    valid = [(name, g, d, r) for name, g, d, r in valid if g is not None]
    if not valid:
        print("  [skip] no trajectory CSVs found")
        return

    n_cols = len(valid)
    titles = [AGENT_LABELS.get(name, name) for name, *_ in valid]
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.12)

    for col, (name, gaps, desired, _) in enumerate(valid, start=1):
        T, N = gaps.shape
        steps = np.arange(T)
        for i in range(N):
            fig.add_trace(go.Scatter(
                x=steps, y=gaps[:, i],
                name=f"Agent {i+1}",
                line=dict(color=AGENT_LINE_COLORS[i % len(AGENT_LINE_COLORS)], width=1.5),
                showlegend=(col == 1),
            ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=steps, y=desired.mean(axis=1),
            name="Desired gap",
            line=dict(color="black", dash="dash", width=1.5),
            showlegend=(col == 1),
        ), row=1, col=col)

        fig.update_xaxes(title_text="Step", row=1, col=col)
        fig.update_yaxes(title_text="Gap (m)", row=1, col=1)

    fig.update_layout(
        title="Best-Episode Gap Trajectory",
        template="plotly_white",
        height=450, width=1000,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=160),
    )
    _save(fig, "fig3_gap_trajectory.png")


def fig4_relv_trajectory(agents):
    # Relative velocity trajectory (string stability indicator)
    valid = [(name, *load_trajectory_csv(name)) for name in agents]
    valid = [(name, g, d, r) for name, g, d, r in valid if g is not None]
    if not valid:
        print("  [skip] no trajectory CSVs found")
        return

    n_cols = len(valid)
    titles = [AGENT_LABELS.get(name, name) for name, *_ in valid]
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.12)

    for col, (name, _, _, relvs) in enumerate(valid, start=1):
        T, N = relvs.shape
        steps = np.arange(T)
        for i in range(N):
            fig.add_trace(go.Scatter(
                x=steps, y=relvs[:, i],
                name=f"Agent {i+1}",
                line=dict(color=AGENT_LINE_COLORS[i % len(AGENT_LINE_COLORS)], width=1.5),
                showlegend=(col == 1),
            ), row=1, col=col)

        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=1, col=col)
        fig.update_xaxes(title_text="Step", row=1, col=col)
        fig.update_yaxes(title_text="Δv (m/s)", row=1, col=1)

    fig.update_layout(
        title="Best-Episode Relative Velocity (Δv = v_pred − v_ego)",
        template="plotly_white",
        height=450, width=1000,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=160),
    )
    _save(fig, "fig4_relv_trajectory.png")


def fig5_attention_weights(agent_name = "attention_mappo"):
    # Attn weight evolution for each agent across time
    attn = load_attention_csv(agent_name)
    if attn is None:
        print(f"  [skip] no attention CSV for {agent_name}")
        return

    T, N, K = attn.shape
    steps = np.arange(T)
    titles = [f"Agent {i+1}" for i in range(N)]
    fig = make_subplots(rows=1, cols=N, subplot_titles=titles,
                        horizontal_spacing=0.06)

    for i in range(N):
        for k in range(K):
            label = NEIGHBOR_LABELS[k] if k < len(NEIGHBOR_LABELS) else f"k={k+1}"
            fig.add_trace(go.Scatter(
                x=steps, y=attn[:, i, k],
                name=label,
                line=dict(color=NEIGHBOR_COLORS[k % len(NEIGHBOR_COLORS)], width=1.5),
                showlegend=(i == 0),
            ), row=1, col=i + 1)

        fig.update_xaxes(title_text="Step", row=1, col=i + 1)
        fig.update_yaxes(title_text="Weight", range=[0, 1], row=1, col=1)

    fig.update_layout(
        title=f"Attention Weights - {AGENT_LABELS.get(agent_name, agent_name)} (Best Episode)",
        template="plotly_white",
        height=400, width=1300,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=180),
    )
    _save(fig, "fig5_attention_weights.png")


def _get_event_windows():
    # Derive per-event step windows from the leader profile config
    leader = SmoothBrakeLeader(n_events=cfg.LEADER_N_EVENTS)
    leader.reset(episode_horizon_duration=cfg.MAX_STEPS * cfg.STEP_LENGTH)
    windows = []
    for start_s, end_s in leader._events:
        start_step = int(start_s / cfg.STEP_LENGTH)
        # 5s response buffer after event ends to capture agent lag
        end_step = min(int((end_s + 5.0) / cfg.STEP_LENGTH), cfg.MAX_STEPS)
        windows.append((start_step, end_step))
    return windows


def fig6_string_stability(agents):
    """
        String stability: whole-episode peak relative vel per agent position, one line per model.
        Left subplot: peak relative vel per agent position (1–5).
        Right subplot: attenuation ratio (peak[i+1] / peak[i]) per consecutive pair.
        Ratio < 1 means the disturbance attenuated — string stable at that vehicle pair.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Peak relative vel per Agent Position", "Attenuation Ratio (agent i+1 / agent i)"],
        horizontal_spacing=0.14,
    )

    positions = None
    pair_labels = None

    for name in agents:
        _, _, relvs = load_trajectory_csv(name)
        if relvs is None:
            print(f"  [skip] no trajectory CSV for {name}")
            continue

        T, N = relvs.shape
        if positions is None:
            positions = list(range(1, N + 1))
            pair_labels = [f"{i+1}→{i+2}" for i in range(N - 1)]

        peak_relv = np.max(np.abs(relvs), axis=0)
        ratios = [float(peak_relv[i + 1] / peak_relv[i]) for i in range(N - 1)]
        color = AGENT_COLORS.get(name, "#888")
        label = AGENT_LABELS.get(name, name)

        fig.add_trace(go.Scatter(
            x=positions, y=peak_relv.tolist(),
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pair_labels, y=ratios,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
            showlegend=False,
        ), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", line_width=1,
                  annotation_text="ratio = 1", annotation_position="right", row=1, col=2)

    fig.update_xaxes(title_text="Agent position", row=1, col=1)
    fig.update_xaxes(title_text="Vehicle pair", row=1, col=2)
    fig.update_yaxes(title_text="Peak relative vel (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Attenuation ratio", row=1, col=2)
    fig.update_layout(
        title="String Stability Analysis - Whole Episode (Best Episode)",
        template="plotly_white",
        height=450, width=1100,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=200),
    )
    _save(fig, "fig6_string_stability.png")


def fig7_attention_gap_correlation(agent_name = "attention_mappo"):
    """
        For each agent: gap error (gap - desired) and the attention weight on the closest predecessor (w0) 
        plotted on dual y-axes over time. Tests whether the attention mechanism responds to spacing errors,
        meaning, if w0 spikes when gap error is large, the weights are faithful control relevant signals
    """
    gaps, desired, _ = load_trajectory_csv(agent_name)
    attn = load_attention_csv(agent_name)

    if gaps is None or attn is None:
        print(f"  [skip] missing trajectory or attention CSV for {agent_name}")
        return

    T, N = gaps.shape
    steps = list(range(T))
    titles = [f"Agent {i + 1}" for i in range(N)]

    fig = make_subplots(
        rows=1, cols=N,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        specs=[[{"secondary_y": True}] * N],
    )

    for i in range(N):
        gap_err = gaps[:, i] - desired[:, i]
        w0 = attn[:, i, 0]

        fig.add_trace(go.Scatter(
            x=steps, y=gap_err.tolist(),
            name="Gap error (m)" if i == 0 else None,
            showlegend=(i == 0),
            line=dict(color="#636EFA", width=1.2),
        ), row=1, col=i + 1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=steps, y=w0.tolist(),
            name="Attn weight w₀" if i == 0 else None,
            showlegend=(i == 0),
            line=dict(color="#EF553B", width=1.2),
        ), row=1, col=i + 1, secondary_y=True)

        fig.update_xaxes(title_text="Step", row=1, col=i + 1)
        fig.update_yaxes(title_text="Gap error (m)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Weight", range=[0, 1], row=1, col=N, secondary_y=True)

    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1,
                  secondary_y=False)

    fig.update_layout(
        title=f"Attention Weight vs Gap Error - {AGENT_LABELS.get(agent_name, agent_name)} (Best Episode)",
        template="plotly_white",
        height=420, width=1400,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=180),
    )
    _save(fig, "fig7_attention_gap_correlation.png")


# Helpers
def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.write_image(path)
    print(f"  Saved {path}")


if __name__ == "__main__":
    agents = [a for a in ["mappo", "attention_mappo"] if os.path.exists(os.path.join(DATA_DIR, f"{a}_eval.csv"))]

    if not agents:
        print("No eval CSVs found in results/data/. Run pipeline/evaluate.py first")
    else:
        print(f"Generating figures for: {agents}")
        fig1_reward_curve(agents)
        fig2_critic_loss(agents)
        fig3_gap_trajectory(agents)
        fig4_relv_trajectory(agents)
        if "attention_mappo" in agents:
            fig5_attention_weights("attention_mappo")
        fig6_string_stability(agents)
        if "attention_mappo" in agents:
            fig7_attention_gap_correlation("attention_mappo")
        print("Done")
