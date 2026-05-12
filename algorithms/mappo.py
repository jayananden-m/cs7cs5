import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import config as cfg


class Actor(nn.Module):
    """Plain MLP policy: local obs (9,) to action (1,) + log_prob"""
    def __init__(self, obs_dim=cfg.LOCAL_OBS_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))


    def forward(self, obs):
        h = self.net(obs)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        return action, dist.log_prob(action)


    def evaluate(self, obs, action):
        h = self.net(obs)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist.log_prob(action), dist.entropy()


class Critic(nn.Module):
    """Centralized MLP value function: global observation (45,) to value (1,)  """
    def __init__(self, obs_dim = cfg.GLOBAL_OBS_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, obs):
        return self.net(obs)


# Rollout Buffer
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.local_obs = []
        self.global_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.size = 0

    
    def push(self, local_obs, global_obs, actions, log_probs, rewards, values, done):
        self.local_obs.append(local_obs)
        self.global_obs.append(global_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(done)
        self.size += 1
    

    def compute_gae(self, last_values):
        # Generalized Advantage Estimation (backwards pass)
        T = self.size
        N = cfg.N_AGENTS
        advantages = np.zeros((T, N, 1), dtype=np.float32)
        returns = np.zeros((T, N, 1), dtype=np.float32)

        rewards = np.stack(self.rewards)
        values = np.stack(self.values)
        dones = np.array(self.dones, dtype=np.float32)
        
        last_gae = np.zeros((N, 1), dtype=np.float32)
        next_vals = last_values

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            r = rewards[t].reshape(N, 1)
            v = values[t]
            delta = r + cfg.GAMMA * next_vals * mask - v
            last_gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * mask * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + v
            next_vals = v
        
        self._advantages = advantages
        self._returns = returns

    

    def get_batches(self, batch_size):
        T, N = self.size, cfg.N_AGENTS
        total = T * N

        lo = np.stack(self.local_obs).reshape(total, cfg.LOCAL_OBS_DIM)
        go = np.tile(np.stack(self.global_obs)[:, None, :], (1, N, 1)).reshape(total, cfg.GLOBAL_OBS_DIM)
        ac = np.stack(self.actions).reshape(total, 1)
        lp = np.stack(self.log_probs).reshape(total, 1)
        adv = self._advantages.reshape(total, 1)
        ret = self._returns.reshape(total, 1)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8) # Normalize advantages
        idx = np.random.permutation(total)
        for start in range(0, total, batch_size):
            b = idx[start:start+batch_size]
            yield {
                "local_obs": torch.tensor(lo[b]),
                "global_obs": torch.tensor(go[b]),
                "actions": torch.tensor(ac[b]),
                "old_log_probs": torch.tensor(lp[b]),
                "advantages": torch.tensor(adv[b]),
                "returns": torch.tensor(ret[b])
            }


# MAPPO Trainer
class MAPPO:
    def __init__(self, env, actor_class=Actor, critic_class=Critic,
                 ppo_epochs=cfg.PPO_EPOCHS, clip_eps=cfg.CLIP_EPS):
        self.env = env
        self.actor = actor_class()
        self.critic = critic_class()
        self.buffer = RolloutBuffer()
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.LR_ACTOR)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.LR_CRITIC)

    
    @torch.no_grad()
    def collect_rollout(self):
        self.buffer.clear()
        local_obs, global_obs, _ = self.env.reset()
        ep_reward, ep_steps = 0.0, 0

        for _ in range(cfg.MAX_STEPS):
            lo_t = torch.tensor(local_obs, dtype=torch.float32)
            go_t = torch.tensor(global_obs, dtype=torch.float32)

            actions, log_probs = self.actor(lo_t)
            values = self.critic(go_t.unsqueeze(0)).expand(cfg.N_AGENTS, 1)

            next_lo, next_go, rewards, done, info = self.env.step(actions.numpy().flatten())
            self.buffer.push(local_obs, global_obs, actions.numpy(), log_probs.numpy(), rewards, values.numpy(), done)

            ep_reward += rewards.sum()
            ep_steps += 1
            local_obs, global_obs = next_lo, next_go

            if done:
                break
        if done:
            last_vals = np.zeros((cfg.N_AGENTS, 1), dtype=np.float32)
        else:
            go_t = torch.tensor(global_obs, dtype=torch.float32)
            last_vals = self.critic(go_t.unsqueeze(0)).expand(cfg.N_AGENTS, 1).numpy()

        self.buffer.compute_gae(last_vals)
        return {
            "ep_reward": float(ep_reward),
            "ep_steps": ep_steps,
            "reason": info.get("reason", "")
        }
    
    
    def update(self):
        """Run PPO_EPOCHS of minibatch updates. Returns averaged losses."""
        total_actor_loss = total_critic_loss = n = 0
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(cfg.BATCH_SIZE):
                lo, go = batch['local_obs'], batch['global_obs']
                old_ac = batch['actions']
                old_lp = batch['old_log_probs']
                adv = batch['advantages']
                ret = batch['returns']

                new_lp, entropy = self.actor.evaluate(lo, old_ac)
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                a_loss = -torch.min(surr1, surr2).mean() - cfg.ENTROPY_COEF * entropy.mean()

                self.actor_optim.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.MAX_GRAD_NORM)
                self.actor_optim.step()

                c_loss = cfg.VALUE_COEF * nn.functional.mse_loss(self.critic(go), ret)
                self.critic_optim.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
                self.critic_optim.step()

                total_actor_loss += a_loss.item()
                total_critic_loss += c_loss.item()
                n += 1
        return {
            "actor_loss": total_actor_loss / max(n, 1),
            "critic_loss": total_critic_loss / max(n, 1)
        }

    
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    
    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
