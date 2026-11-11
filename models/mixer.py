import torch
import torch.nn as nn

class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs: torch.Tensor) -> torch.Tensor:
        return torch.sum(agent_qs, dim=-1, keepdim=True)