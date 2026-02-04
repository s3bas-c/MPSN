import MoE_model as MoE
import MPSN_model as MPSN
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

learning_rate = 0.001

_MPSN_model = MPSN.Router(2, 1, 128, 4, 64)
_MPSN_model.load_state_dict(torch.load("MPSN_model.pt"))
_MPSN_optim = torch.optim.Adam(_MPSN_model.parameters(), lr=learning_rate)

_MoE_model = MoE.Router(2, 1, 128, 2, 64)
_MoE_model.load_state_dict(torch.load("MoE_model.pt"))
_MoE_optim = torch.optim.Adam(_MoE_model.parameters(), lr=learning_rate)

class shared_dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = torch.from_numpy(np.load("Inputs.npy")).float()
        self.outputs = torch.from_numpy(np.load("Outputs.npy")).float()
    def __len__(self):
        return len(self.outputs)
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

dataset = shared_dataset()

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = int(0.15 * total_size)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train = True
epochs = 20

if train:
    for e in range(epochs):
        for batch_x, batch_y in train_loader:
            #MPSN
            _MPSN_optim.zero_grad()

            y = _MPSN_model(batch_x)
            mpsn_loss = F.mse_loss(y, batch_y)
            mpsn_loss.backward()

            _MPSN_optim.step()

            #MoE
            _MoE_optim.zero_grad()

            y = _MoE_model(batch_x)
            moe_loss = F.mse_loss(y, batch_y)
            moe_loss.backward()

            _MoE_optim.step()
