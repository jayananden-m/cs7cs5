# Usage:
#   python -m analysis.plot_leader_profile
#   python -m analysis.plot_leader_profile --seed 7 --events 3

import argparse
import numpy as np
import plotly.graph_objects as go

import config as cfg
from env.leader_profile import SmoothBrakeLeader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--events", type=int, default=1)
    args = parser.parse_args()

    episode_duration = cfg.MAX_STEPS * cfg.STEP_LENGTH
    leader = SmoothBrakeLeader(n_events=args.events)
    leader.reset(episode_horizon_duration=episode_duration, seed=args.seed)

    times = np.arange(0, episode_duration, cfg.STEP_LENGTH)
    speeds = [leader.desired_speed(float(t)) for t in times]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=speeds,
        mode="lines",
        name="Leader speed",
        line=dict(color="#636EFA", width=2),
    ))

    fig.add_hline(y=leader.cruise_speed, line_dash="dot", line_color="grey",
                  line_width=1, annotation_text="cruise", annotation_position="right")
    fig.add_hline(y=leader.brake_speed, line_dash="dot", line_color="#EF553B",
                  line_width=1, annotation_text="brake", annotation_position="right")

    for i, (start, end) in enumerate(leader._events):
        fig.add_vrect(x0=start, x1=end, fillcolor="#EF553B", opacity=0.08,
                      line_width=0, annotation_text=f" event {i+1}",
                      annotation_position="top left")

    fig.update_layout(
        title=f"Leader Speed Profile (seed={args.seed}, {args.events} event(s))",
        xaxis_title="Time (s)",
        yaxis_title="Speed (m/s)",
        template="plotly_white",
        height=400, width=900,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=160),
    )

    out_dir = "results/figures"
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "leader_profile.png")
    fig.write_image(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
