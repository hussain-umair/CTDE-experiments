import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os  
import numpy as np 
import torch 

os.environ["SC2PATH"] = os.path.expanduser("~/vdn_experiment/StarCraftII")

from smac.env import StarCraft2Env

from config import Config
from qmix.learner import QMIXLearner 
from utils.replay_buffer import EpisodeReplayBuffer 
from utils.plotter import TrainingPlotter

def train(): 
    cfg = Config() 
    # 1. Initialize SMAC Environment 
    env = StarCraft2Env(map_name=cfg.map_name)
    env_info = env.get_env_info()
    
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    obs_dim = env_info["obs_shape"] 
    state_dim = env_info["state_shape"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training VDN on SMAC map '{cfg.map_name}' using device: {device}")

    # 2. Initialize QMIX Learner & Buffer 
    learner = QMIXLearner(
        n_agents=n_agents, 
        obs_dim=obs_dim,
        n_actions=n_actions,
        state_dim=state_dim, 
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        mixing_embed_dim=cfg.mixing_embed_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        device=device
    )

    buffer = EpisodeReplayBuffer(cfg.buffer_size)
    plotter = TrainingPlotter(log_dir="logs", filename="qmix_metrics.csv")

    epsilon = cfg.epsilon_start 
    epsilon_decrement = (cfg.epsilon_start - cfg.epsilon_min) / cfg.epsilon_anneal_steps

    t_env = 0 
    episode_num = 0

    while t_env < cfg.total_steps:
        episode_num += 1 
        env.reset()
        terminated = False
        episode_reward = 0

        ep_obs, ep_state, ep_actions, ep_rewards, ep_terminated, ep_avail = [], [], [], [], [], []

        hidden_eval = learner.agent_eval.init_hidden(n_agents).to(device)
        last_action = torch.zeros(n_agents, n_actions).to(device)

        while not terminated: 
            obs = env.get_obs() # List of n_agents arrays 
            state = env.get_state() # Global state array 
            avail_actions = env.get_avail_actions() # [n_agents, n_actions]

            # Construct input vector: [obs, last_action, agent_id_one_hot] 
            inputs = [] 
            for i in range(n_agents):
                agent_id_one_hot = torch.zeros(n_agents)
                agent_id_one_hot[i] = 1.0
                inp = torch.cat([torch.tensor(obs[i], dtype=torch.float32), last_action[i].cpu(), agent_id_one_hot]) 
                inputs.append(inp)

            inputs = torch.stack(inputs).to(device) # [n_agents, input_shape]

            # Select actions with GRU agent 
            with torch.no_grad():
                q_vals, hidden_eval = learner.agent_eval(inputs, hidden_eval)

            # Epsilon-greedy action selection 
            actions = [] 
            for i in range(n_agents): 
                avail = np.array(avail_actions[i])
                avail_indices = np.flatnonzero(avail)
                
                if len(avail_indices) == 0:
                    actions.append(0) # Default no-op if dead 
                elif np.random.rand() < epsilon:
                    actions.append(int(np.random.choice(avail_indices))) 
                else: 
                    q_avail = q_vals[i].cpu().numpy().copy()
                    q_avail[avail == 0] = -1e10 
                    actions.append(int(q_avail.argmax()))

            # Step environment 
            reward, terminated, _ = env.step(actions)
            episode_reward += reward 
            t_env += 1

            # Update epsilon 
            epsilon = max(cfg.epsilon_min, epsilon - epsilon_decrement)

            # Update last action one-hot tensor 
            last_action = torch.zeros(n_agents, n_actions).to(device)
            for i, act in enumerate(actions):
                last_action[i, act] = 1.0

            # Store step data 
            ep_obs.append(np.array(inputs.cpu()))
            ep_state.append(state)
            ep_actions.append(actions)
            ep_rewards.append([reward])
            ep_terminated.append([float(terminated)])
            ep_avail.append(avail_actions)

        # Store complete episode in Replay Buffer 
        buffer.push({ 
            "obs": np.array(ep_obs),
            "state": np.array(ep_state),
            "actions": np.array(ep_actions),
            "rewards": np.array(ep_rewards),
            "terminated": np.array(ep_terminated),
            "avail_actions": np.array(ep_avail) 
        })

        # Train on sampled batch once buffer has enough samples 
        loss = 0.0 
        if len(buffer) >= cfg.batch_size:
            loss = learner.train_batch(buffer.sample(cfg.batch_size))

        # Target network update 
        if episode_num % cfg.target_update_interval == 0:
            learner.update_targets()

        # Logging 
        if episode_num % cfg.log_interval == 0:
            print(f"Episode: {episode_num:5d} | Timestep: {t_env:7d} | Return: {episode_reward:5.2f} | Loss: {loss:6.4f} | Epsilon: {epsilon:4.3f}") 
            plotter.log(episode_num, t_env, episode_reward, loss, epsilon)
            plotter.plot(save_path="logs/qmix_training_metrics.png")

    env.close() 
if __name__ == "__main__":
    train()
