import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Expert class for defining experts

class Expert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)  
        x = self.l2(x)
        x = F.relu(x)  
        return x

class Router(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, expert_amount, router_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size) # Shared input for all experts
        self.output_layer = nn.Linear(hidden_size, output_size) # Shared output for all experts

        self.expert_list = nn.ModuleList(Expert(hidden_size=hidden_size) for _ in range(expert_amount)) # Expert list for routing

        # Router layers
        self.router_l1 = nn.Linear(hidden_size, router_size) 
        self.router_l2 = nn.Linear(router_size, expert_amount)

        self.expert_amount = expert_amount
    
    def forward(self, x):
        # Shared input
        x = self.input_layer(x)
        router_x = self.router_l1(x)
        router_x = F.relu(router_x)
        router_x = self.router_l2(router_x)

        # Softmax for selecting weights
        probs = F.softmax(router_x, dim=-1)

        out = torch.zeros_like(x)
        # Applying weights
        for e in range(self.expert_amount):
            out += self.expert_list[e](x) * probs[:, e:e+1]
        out = self.output_layer(out)
        # Shared output
        return out