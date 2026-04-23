import sys

import torch

sys.path.insert(0, 'src')
from model import TwixNet

net = TwixNet(num_filters=64, num_blocks=8)
torch.save(net, 'models/v0.pt')

num_params = sum(p.numel() for p in net.parameters())
print(f"Saved models/v0.pt  ({num_params:,} parameters)")
