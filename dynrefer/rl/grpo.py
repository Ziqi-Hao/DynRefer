import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class ViewSelector(nn.Module):
    """Simple policy network for selecting nested views."""

    def __init__(self, state_dim: int, hidden_dim: int, num_views: int, num_select: int):
        super().__init__()
        self.num_views = num_views
        self.num_select = num_select
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_views),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

    def log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the log probability of selecting given actions."""
        probs = self.forward(state)
        act_probs = probs.gather(-1, actions)
        return act_probs.log().sum(-1)

    def sample(self, state: torch.Tensor, g: int):
        """Sample G different view combinations."""
        probs = self.forward(state)
        actions = []
        log_probs = []
        for _ in range(g):
            idx = torch.multinomial(probs, self.num_select, replacement=False)
            actions.append(idx)
            log_probs.append(self.log_prob(state, idx))
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        return actions, log_probs

@dataclass
class GRPOConfig:
    g: int = 8
    clip_eps: float = 0.2
    lr: float = 1e-4

class GRPOTrainer:
    """Minimal implementation of Group-Relative Policy Optimization."""

    def __init__(self, policy: ViewSelector, env, cfg: GRPOConfig = GRPOConfig()):
        self.policy = policy
        self.env = env
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def train_batch(self, batch):
        state = self.env.get_state(batch)
        actions, old_logp = self.policy.sample(state, self.cfg.g)
        rewards = []
        for a in actions:
            r = self.env.evaluate(batch, a)
            rewards.append(r)
        rewards = torch.stack(rewards)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        new_logp = torch.stack([self.policy.log_prob(state, a) for a in actions])
        ratio = (new_logp - old_logp.detach()).exp()
        obj = ratio * adv
        obj_clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
        loss = -(torch.min(obj, obj_clipped)).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), rewards.mean().item()

class DynReferEnv:
    """Wrapper of a frozen DynRefer model for RL training."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def get_state(self, sample):
        return sample["state"]

    def evaluate(self, sample, views):
        sample = dict(sample)
        sample["selected_views"] = views
        with torch.no_grad():
            output = self.model.predict_answers(sample)
        return torch.tensor(output[0].get("score", 0.0))
