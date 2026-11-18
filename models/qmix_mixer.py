import torch
import torch.nn as nn
import torch.nn.functional as F

class QMIXMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32, hypernet_embed_dim: int = 64):
        super(QMIXMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, self.n_agents * self.embed_dim)
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, self.embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = agent_qs.shape
        
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(bs, seq_len, 1)

        return q_tot