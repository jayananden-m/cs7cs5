# Attention Actor with Positional Encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


import config as cfg

class AttentionActor(nn.Module):
    """
        Policy with learned attention over K neighbors.

        Instead of treating the flat (9,) observation as unstructured, reshapes
        to (K, 3) — one row per neighbor — and learns which neighbor to focus on.

        Positional encoding (Vaswani et al. 2017, learned variant):
            Each neighbor embedding receives a learned position vector before
            scoring. Index 0 = closest predecessor, K-1 = furthest.
            Without PE, attention weights are near-uniform (~1/K) and provide
            no useful gradient signal for platoon following.

        Flow:
            obs (9,)
            -> reshape (K, 3)
            -> embed: Linear(3 -> embed_dim) + tanh     -> (K, embed_dim)
            -> add pos_embed[j]                         -> (K, embed_dim)
            -> score: Linear(embed_dim -> 1) -> softmax -> (K,)
            -> context: weighted sum                    -> (embed_dim,)
            -> MLP -> mean (1,)
    """
    N_FEATURES = 3 # gap, rel_v, speed

    def __init__(self, k = cfg.K_NEIGHBORS, embed_dim = 32, hidden = 64):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim
        
        self.neighbor_embed = nn.Linear(self.N_FEATURES, embed_dim)
        self.pos_embed = nn.Embedding(k, embed_dim)
        self.attention_score = nn.Linear(embed_dim, 1, bias=False)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

        # Zero-init score network so initial attention is uniform (≈1/K per neighbour).
        # Prevents seed-sensitive coherent bad policies at random initialisation.
        nn.init.zeros_(self.attention_score.weight)

        
    
    def _attend(self, obs):
        # Compute attended context vector from flat obs
        batch_shape = obs.shape[:-1]
        x = obs.view(*batch_shape, self.k, self.N_FEATURES)
        
        embeddings = torch.tanh(self.neighbor_embed(x))
        pos_ids = torch.arange(self.k, device=obs.device)
        embeddings = embeddings + self.pos_embed(pos_ids)

        scores = self.attention_score(embeddings).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)

        context = (attn_weights.unsqueeze(-1) * embeddings).sum(dim=-2)
        return context, attn_weights
    

    def forward(self, obs):
        context, _ = self._attend(obs)
        h = self.net(context)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        
        return action, dist.log_prob(action)

    
    def evaluate(self, obs, action):
        context, _ = self._attend(obs)
        h = self.net(context)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        return dist.log_prob(action), dist.entropy()
    

    def get_attention_weights(self, obs):
        # Return attn weights for visualization
        with torch.no_grad():
            _, weights = self._attend(obs)
        
        return weights