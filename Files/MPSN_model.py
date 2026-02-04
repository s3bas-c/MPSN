import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

# Module class for selecting layers

class Module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        return x

class Router(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, module_amount, router_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.module_list = nn.ModuleList(Module(hidden_size=hidden_size) for _ in range(module_amount)) # Module list for routing

        self.router_l1 = nn.Linear(hidden_size, router_size)
        self.router_l2 = nn.Linear(router_size, module_amount)

        self.module_amount = module_amount
        self.cycle_projection = nn.Linear(1, hidden_size)

    def forward(self, x, max_cycles):
        x = self.input_layer(x)
        original_x = x # Keep original x so router can rely on cycle and build a path, instead of getting confused with changing input

        for c in range(max_cycles):
            batch_size = x.size(0)

            # Use of cycle tensor to define iteration

            cycle = torch.full((batch_size, 1), c, dtype=torch.float32)
            cycle_proj = self.cycle_projection(cycle)
            
            router_x = original_x * cycle_proj

            router_x = self.router_l1(router_x)
            router_x = F.relu(router_x)
            router_x = self.router_l2(router_x)

            probs = F.softmax(router_x, dim=-1)

            out = torch.zeros_like(x)
            for m in range(self.module_amount):
                out += self.module_list[m](x) * probs[:, m:m+1] # Apply soft routing to x
            x = out # Set x to new value since its iterative

        x = self.output_layer(x)
        return x