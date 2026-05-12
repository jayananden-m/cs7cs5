# Usage:
#   python -m pipeline.train --agent mappo --episodes 500
#   python -m pipeline.train --agent attention_mappo --episodes 500 --seed 42

import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from env.sumo_env import SumoEnv
from algorithms.mappo import MAPPO, Actor, Critic
from algorithms.attention_mappo import AttentionActor


AGENT_REGISTRY = {
    "mappo": {
        "actor": Actor,
        "critic": Critic,
        # Original PPO settings — MLP actor doesn't need conservative clipping
    },
    "attention_mappo": {
        "actor": AttentionActor,
        "critic": Critic,
        # Conservative PPO to prevent overshooting with structured attention init
        "ppo_epochs": 5,
        "clip_eps": 0.1,
    }
}


def train(agent_name, n_episodes, save_every, gui, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Available: {list(AGENT_REGISTRY)}")

    classes = AGENT_REGISTRY[agent_name]
    env = SumoEnv(gui=gui)
    ppo_kwargs = {k: v for k, v in classes.items() if k in ("ppo_epochs", "clip_eps")}
    agent = MAPPO(env, actor_class=classes["actor"], critic_class=classes["critic"], **ppo_kwargs)

    model_dir = os.path.join("results", "models", agent_name)
    log_dir = os.path.join("results", "logs",   agent_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", agent_name))
    best_reward_per_step = float("-inf")

    for ep in range(1, n_episodes + 1):
        rollout = agent.collect_rollout()
        losses = agent.update()

        ep_reward = rollout["ep_reward"]
        ep_steps = rollout["ep_steps"]
        reason = rollout["reason"]
        reward_per_step = ep_reward / max(ep_steps, 1)

        writer.add_scalar("reward/episode",  ep_reward, ep)
        writer.add_scalar("reward/per_step", reward_per_step, ep)
        writer.add_scalar("episode/steps", ep_steps, ep)
        writer.add_scalar("loss/actor", losses["actor_loss"],  ep)
        writer.add_scalar("loss/critic", losses["critic_loss"], ep)

        print(
            f"[{agent_name}] Ep {ep:4d} | "
            f"R={ep_reward:8.1f} | steps={ep_steps:4d} | end={reason or '-'} | "
            f"a_loss={losses['actor_loss']:.4f} | c_loss={losses['critic_loss']:.4f}"
        )

        if reward_per_step > best_reward_per_step:
            best_reward_per_step = reward_per_step
            agent.save(os.path.join(model_dir, "best.pt"))

        if ep % save_every == 0:
            agent.save(os.path.join(model_dir, f"ep{ep}.pt"))

    agent.save(os.path.join(model_dir, "final.pt"))
    env.close()
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="mappo")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    train(args.agent, args.episodes, args.save_every, args.gui, args.seed)

