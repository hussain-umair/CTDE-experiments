import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch 
import torch.nn as nn 
import torch.optim as optim 
from models.agent import RNNAgent 
from models.qmix_mixer import QMIXMixer

class QMIXLearner:
    def __init__(
        self, 
        n_agents: int, 
        obs_dim: int, 
        n_actions: int, 
        state_dim: int, 
        rnn_hidden_dim: int = 64, 
        mixing_embed_dim: int = 32, 
        lr: float = 5e-4, 
        gamma: float = 0.99, 
        device: str = "cpu" 
    ):
        self.n_agents = n_agents 
        self.n_actions = n_actions 
        self.state_dim = state_dim 
        self.gamma = gamma 
        self.device = device

        self.input_shape = obs_dim + n_actions + n_agents

        self.agent_eval = RNNAgent(self.input_shape, rnn_hidden_dim, n_actions).to(device)
        self.agent_target = RNNAgent(self.input_shape, rnn_hidden_dim, n_actions).to(device)
        self.agent_target.load_state_dict(self.agent_eval.state_dict())

        self.mixer_eval = QMIXMixer(n_agents, state_dim, mixing_embed_dim).to(device)
        self.mixer_target = QMIXMixer(n_agents, state_dim, mixing_embed_dim).to(device)
        self.mixer_target.load_state_dict(self.mixer_eval.state_dict())

        self.params = list(self.agent_eval.parameters()) + list(self.mixer_eval.parameters()) 
        self.optimizer = optim.RMSprop(self.params, lr=lr, alpha=0.99, eps=1e-5)

    def update_targets(self):
        self.agent_target.load_state_dict(self.agent_eval.state_dict())
        self.mixer_target.load_state_dict(self.mixer_eval.state_dict())

    def train_batch(self, batch: dict) -> float: 
        obs = batch["obs"].to(self.device) # [bs, seq_len, n_agents, obs_dim] 
        actions = batch["actions"].long().to(self.device) # [bs, seq_len, n_agents] 
        rewards = batch["rewards"].to(self.device) # [bs, seq_len, 1] 
        terminated = batch["terminated"].to(self.device) # [bs, seq_len, 1] 
        avail_actions = batch["avail_actions"].to(self.device) # [bs, seq_len, n_agents, n_actions] 
        states = batch["state"].to(self.device) # [bs, seq_len, state_dim]

        if actions.dim() == 3: 
            actions = actions.unsqueeze(-1)

        batch_size, seq_len, _, _ = obs.shape

        q_evals = [] 
        hidden_eval = self.agent_eval.init_hidden(batch_size * self.n_agents).to(self.device)

        for t in range(seq_len): 
            obs_t = obs[:, t].reshape(-1, self.input_shape)
            q_t, hidden_eval = self.agent_eval(obs_t, hidden_eval) 
            q_evals.append(q_t.reshape(batch_size, self.n_agents, self.n_actions))

        q_evals = torch.stack(q_evals, dim=1) 
        chosen_q_evals = torch.gather(q_evals, dim=-1, index=actions).squeeze(-1)

        q_targets = [] 
        hidden_target = self.agent_target.init_hidden(batch_size * self.n_agents).to(self.device)

        for t in range(seq_len): 
            obs_t = obs[:, t].reshape(-1, self.input_shape) 
            q_t, hidden_target = self.agent_target(obs_t, hidden_target) 
            q_targets.append(q_t.reshape(batch_size, self.n_agents, self.n_actions))

        q_targets = torch.stack(q_targets, dim=1)

        # Mask unavailable actions on next states (t = 1 to seq_len) 
        mac_out_eval_next = q_evals[:, 1:].clone() 
        mac_out_eval_next[avail_actions[:, 1:] == 0] = -1e10 
        cur_max_actions = mac_out_eval_next.argmax(dim=-1, keepdim=True)

        target_max_q_vals = torch.gather(q_targets[:, 1:], dim=-1, index=cur_max_actions).squeeze(-1)

        q_tot_eval = self.mixer_eval(chosen_q_evals[:, :-1], states[:, :-1]) 
        q_tot_target = self.mixer_target(target_max_q_vals, states[:, 1:])

        targets = rewards[:, :-1] + self.gamma * (1 - terminated[:, :-1]) * q_tot_target.detach() 
        td_error = q_tot_eval - targets 
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, max_norm=10.0) 
        self.optimizer.step()

        return loss.item()









