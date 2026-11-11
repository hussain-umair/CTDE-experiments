import os 
import csv 
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TrainingPlotter: 
    """ Logs episode metrics (reward, loss, epsilon) to CSV and renders publication-ready plots. """ 
    
    def __init__(self, log_dir: str = "logs", filename: str = "training_metrics.csv"):
        self.log_dir = log_dir 
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)

        # Initialize CSV file header 
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f) 
            writer.writerow(["episode", "timestep", "return", "loss", "epsilon"])
    
    def log(self, episode: int, timestep: int, episode_return: float, loss: float, epsilon: float):
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f) 
            writer.writerow([episode, timestep, episode_return, loss, epsilon])

    def plot(self, save_path: str = "logs/vdn_training_metrics.png", window: int = 15):
        if not os.path.exists(self.csv_path):
            return
        
        data = np.genfromtxt(self.csv_path, delimiter=',', skip_header=1)
        if len(data) == 0:
            return 
        if data.ndim == 1:
            data = data.reshape(1, -1)
        episodes = data[:, 0]
        returns = data[:, 2]
        losses = data[:, 3]
        epsilons = data[:, 4]
        
        # Edge-padded moving average to eliminate boundary artifacts 
        def smooth(arr, w):
            if len(arr) < w:
                return arr
            padded = np.pad(arr, (w // 2, w // 2), mode='edge')
            smoothed = np.convolve(padded, np.ones(w) / w, mode='valid')
            return smoothed[:len(arr)]
        
        smooth_returns = smooth(returns, window)
        smooth_losses = smooth(losses, window)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # 1. Return Plot
        ax1.plot(episodes, returns, color='#0072b2', alpha=0.25, linewidth=1, label="Raw Return")
        ax1.plot(episodes, smooth_returns, color='#0072b2', linewidth=2.5, label=f"{window}-Ep Moving Avg")
        ax1.axhline(20.0, color='#8a8a8a', linestyle='--', label="Max Return (20.0)")
        ax1.set_ylabel("Return / Reward")
        ax1.set_title("Episode Return (Reward Progression)")
        ax1.legend(loc="lower right", frameon=False) 
        ax1.grid(True, axis='y', alpha=0.3)

        # 2. TD Loss Plot 
        ax2.plot(episodes, losses, color='#d55e00', alpha=0.25, linewidth=1, label="Raw TD Loss")
        ax2.plot(episodes, smooth_losses, color='#d55e00', linewidth=2.5, label=f"{window}-Ep Moving Avg")
        ax2.set_ylabel("TD Loss")
        ax2.set_title("Temporal Difference (TD) Loss")
        ax2.legend(loc="upper right", frameon=False)
        ax2.grid(True, axis='y', alpha=0.3)

        # 3. Epsilon Plot 
        ax3.plot(episodes, epsilons, color='#e69f00', linewidth=2.5)
        ax3.set_ylabel("Epsilon (ε)") 
        ax3.set_xlabel("Episode")
        ax3.set_ylim(0, 1.05)
        ax3.axhline(0.05, color='#888888', linestyle='--', label="ε_min (0.05)")
        ax3.set_title("Exploration Rate (Epsilon Annealing)")
        ax3.grid(True, axis='y', alpha=0.3)
        plt.suptitle("VDN Training Dynamics on SMAC", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()