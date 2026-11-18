import torch
import torch.nn as nn
import torch.optim as optim
from models.agent import RNNAgent
from models.vdn_mixer import VDNMixer

class VDNLearner:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        rnn_hidden_dim: int=64,
        lr: float=5e-4,
        gamma: float=0.99,
        device: str="cpu"
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device

        self.input_shape = obs_dim + n_actions + n_agents # vector

        self.agent_eval = RNNAgent(self.input_shape, rnn_hidden_dim, n_actions).to(device)
        self.agent_target = RNNAgent(self.input_shape, rnn_hidden_dim, n_actions).to(device)
        self.agent_target.load_state_dict(self.agent_eval.state_dict())

        self.mixer_eval = VDNMixer().to(device)
        self.mixer_target = VDNMixer().to(device)

        self.params = list(self.agent_eval.parameters()) + list(self.mixer_eval.parameters())
        self.optimizer = optim.RMSprop(self.params, lr=lr, alpha=0.99, eps=1e-5)

    def update_targets(self):
        self.agent_target.load_state_dict(self.agent_eval.state_dict())
        self.mixer_target.load_state_dict(self.mixer_eval.state_dict())

    def train_batch(self, batch: dict) -> float:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].long().to(self.device)
        rewards = batch["rewards"].to(self.device)
        terminated = batch["terminated"].to(self.device)
        avail_actions = batch["avail_actions"].to(self.device)

        if actions.dim() == 3:
            actions = actions.unsqueeze(-1)
            
        batch_size, seq_len, _, _ = obs.shape

        # 1. Compute evaluation Q_tot
        q_evals = []
        hidden_eval = self.agent_eval.init_hidden(batch_size * self.n_agents).to(self.device)

        for t in range(seq_len):
            obs_t = obs[:, t].reshape(-1, self.input_shape)
            q_t, hidden_eval = self.agent_eval(obs_t, hidden_eval)
            q_evals.append(q_t.reshape(batch_size, self.n_agents, self.n_actions))

        q_evals = torch.stack(q_evals, dim=1)
        chosen_q_evals = torch.gather(q_evals, dim=-1, index=actions).squeeze(-1)

        # 2.Compute target Q_tot with SMAC action masking
        q_targets = []
        hidden_target = self.agent_target.init_hidden(batch_size * self.n_agents).to(self.device)

        for t in range(seq_len):
            obs_t = obs[:,t].reshape(-1, self.input_shape)
            q_t, hidden_target = self.agent_target(obs_t, hidden_target)
            q_targets.append(q_t.reshape(batch_size, self.n_agents, self.n_actions))

        q_targets = torch.stack(q_targets, dim=1)

        # Double-DQN action selection on (t = 1 to seq_len) 
        mac_out_eval_next = q_evals[:, 1:].clone()
        mac_out_eval_next[avail_actions[:, 1:] == 0] = -1e10
        cur_max_actions = mac_out_eval_next.argmax(dim=-1, keepdim=True)
        
        target_max_q_vals = torch.gather(q_targets[:, 1:], dim=-1, index=cur_max_actions).squeeze(-1)

        q_tot_eval = self.mixer_eval(chosen_q_evals[:, :-1])
        q_tot_target = self.mixer_target(target_max_q_vals)

        # 3. Calculate TD loss and optimize
        targets = rewards[:, :-1] + self.gamma * (1 - terminated[:, :-1]) * q_tot_target.detach()
        td_error = q_tot_eval - targets
        loss = (td_error ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, max_norm=10.0)
        self.optimizer.step()

        return loss.item()
        
    

        






        