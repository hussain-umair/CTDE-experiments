import random
import numpy as np
import torch

class EpisodeReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.memory = []

    def push(self, episode_dict: dict):
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append(episode_dict)

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.memory, batch_size)

        max_ep_len = max(len(ep["obs"]) for ep in batch)
        
        batch_dict = {}
        for k in batch[0].keys():
            padded_list = []
            for ep in batch:
                arr = ep[k]
                pad_len = max_ep_len - len(arr)
                if pad_len > 0:
                    pad_shape = (pad_len,) + arr.shape[1:]
                    if k == "terminated":
                        padding = np.ones(pad_shape, dtype=arr.dtype)
                    else:
                        padding = np.zeros(pad_shape, dtype=arr.dtype)
                    padded_arr = np.concatenate([arr, padding], axis=0)
                else:
                    padded_arr = arr
                padded_list.append(padded_arr) 
            batch_dict[k] = torch.tensor(np.array(padded_list), dtype=torch.float32)
        return batch_dict

    def __len__(self):
        return len(self.memory)