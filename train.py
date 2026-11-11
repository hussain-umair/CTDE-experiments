import os
import sys
import numpy as np
import torch

os.environ["SC2PATH"] = os.path.expanduser("~/vdn_experiment/StarCraftII")

from smac.env import StarCraft2Env
from config import Config
from learner import VDNLearner
from utils.replay_buffer import EpisodeReplayBuffer
from utils.plotter import TrainingPlotter

def train():
    cfg = Config()
    env = StarCraft2Env(map_name=cfg.map_name)
    env_info = env.get_env_info()

    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training VDN on SMAC map '{cfg.map_name}' using device: {device}")

    learner = VDNLearner(
        n_agents=n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        device=device
    )

    buffer = EpisodeReplayBuffer(buffer_size=cfg.buffer_size)
    plotter = TrainingPlotter()

    epsilon = cfg.epsilon_start
    t_env = 0
    episode_num = 0

    while t_env < cfg.total_steps:
        env.reset()
        terminated = False
        episode_reward = 0

        ep_obs, ep_actions, ep_rewards, ep_terminated, ep_avail_actions = [], [], [], [], []
        last_action = np.zeros((n_agents, n_actions))
        hidden = learner.agent_eval.init_hidden(n_agents).to(device)

        while not terminated:
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            inputs = []
            for agent_id in range(n_agents):
                agent_id_onehot = np.zeros(n_agents)
                agent_id_onehot[agent_id] = 1.0
                inp = np.concatenate([obs[agent_id], last_action[agent_id], agent_id_onehot])
                inputs.append(inp)

            inputs_t = torch.tensor(np.array(inputs), dtype=torch.float32).to(device)
            q_vals, hidden = learner.agent_eval(inputs_t, hidden)
            q_vals = q_vals.cpu().detach().numpy()

            actions = []
            for agent_id in range(n_agents):
                avail = np.array(avail_actions[agent_id])
                if np.random.rand() < epsilon:
                    avail_indices = np.flatnonzero(avail)
                    if len(avail_indices) > 0:
                        action = int(np.random.choice(avail_indices))
                    else:
                        action = 0
                else:
                    q_vals_agent = q_vals[agent_id].copy()
                    q_vals_agent[avail == 0] = -1e10
                    action = int(np.argmax(q_vals_agent))
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            t_env += 1
            episode_reward += reward

            last_action = np.zeros((n_agents, n_actions))
            for i, act in enumerate(actions):
                last_action[i, act] = 1.0

            ep_obs.append(inputs)
            ep_actions.append(np.expand_dims(actions, axis=-1))
            ep_rewards.append([reward])
            ep_terminated.append([float(terminated)])
            ep_avail_actions.append(avail_actions)

            epsilon = max(cfg.epsilon_min, epsilon - (cfg.epsilon_start - cfg.epsilon_min) / cfg.epsilon_anneal_steps)

        ep_dict = {
            "obs": np.array(ep_obs),
            "actions": np.array(ep_actions),
            "rewards": np.array(ep_rewards),
            "terminated": np.array(ep_terminated),
            "avail_actions": np.array(ep_avail_actions),
        }
        buffer.push(ep_dict)
        episode_num += 1

        if len(buffer) >= cfg.batch_size:
            loss = learner.train_batch(buffer.sample(cfg.batch_size))

            if episode_num % cfg.target_update_interval == 0:
                learner.update_targets()

            if episode_num % cfg.log_interval == 0:
                plotter.log(episode_num, t_env, episode_reward, loss, epsilon)
                plotter.plot()
                print(f"Episode: {episode_num:>4d} | Timestep: {t_env:>6d} | Return: {episode_reward:>6.2f} | Loss {loss:>6.4f} | Epsilon: {epsilon:>5.3f}")

    env.close()

if __name__ == "__main__":
    train()
            
                
                







            