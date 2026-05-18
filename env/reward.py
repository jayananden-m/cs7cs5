# Reward function — Chen et al. (2023), arXiv:2308.02345
# r = -w1*(gap_err/GAP_NORM)^2 - w2*(rel_v/RELV_NORM)^2
#     -w3*(a/A_MAX)^2 - w4*ReLU(2*hs - gap)^2 / (2*hs)^2

import numpy as np
import config as cfg


def compute_reward(gaps, rel_velocities, accelerations, speeds, done, termination_reason):
    """
        Compute per-agent reward for one timestep

        Args:
            gaps: bumper-to-bumper gap to predecessor
            rel_velocities: own speed minus predecessor speed
            accelerations: current acceleration of each agent
            speeds: current speed of each agent
            done: whether the episode ended this step
            termination_reason: collision or runaway or time_up or ""

        Returns:
            rewards: (N,) per-agent reward
    """
    desired_gaps = cfg.D0 + cfg.TAU * speeds  # IDM desired gap

    # Normalised errors
    gap_err_norm = (gaps - desired_gaps) / cfg.GAP_NORM
    relv_err_norm = rel_velocities / cfg.RELV_NORM
    accel_norm = accelerations / cfg.A_MAX

    # Squared normalised penalties (Chen et al. 2023)
    r_spacing = -cfg.W_SPACING * gap_err_norm ** 2
    r_velocity = -cfg.W_VELOCITY * relv_err_norm ** 2
    r_accel = -cfg.W_ACCEL * accel_norm ** 2

    # ReLU safety penalty — active only when gap < 2 * SAFETY_GAP
    safety_violation = np.maximum(2 * cfg.SAFETY_GAP - gaps, 0.0) / (2 * cfg.SAFETY_GAP)
    r_safety = -cfg.W_SAFETY * safety_violation ** 2

    rewards = r_spacing + r_velocity + r_accel + r_safety

    if done:
        if termination_reason == "collision":
            rewards -= cfg.W_COLLISION
        elif termination_reason == "runaway":
            rewards -= cfg.W_RUNAWAY

    return rewards.astype(np.float32)

