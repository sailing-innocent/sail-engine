import torch 
import torch.nn as nn 

from typing import List 
# used for mnist helloworld
class ReLUMLP(nn.Module):
    def __init__(self, 
        input_channels: int = 28 * 28,
        output_channels: int = 10, 
        depth: int = 0,
        hidden_channels: List[int] = []):
        super().__init__()

        if depth < 0:
            raise ValueError("depth must be positive")

        assert len(hidden_channels) == depth, "hidden_channels must be the same as depth"

        self.depth = depth
        if depth == 0:
            self.linear_stack = nn.ModuleList([nn.Linear(input_channels, output_channels)])
        elif depth == 1:
            self.linear_stack = nn.ModuleList(
                [nn.Linear(input_channels, hidden_channels[0])] + 
                [nn.Linear(hidden_channels[0], output_channels)]
            )
        else:
            self.linear_stack = nn.ModuleList(
                [nn.Linear(input_channels, hidden_channels[0])] + 
                [nn.Linear(hidden_channels[i], hidden_channels[i+1]) for i in range(depth - 1)] + 
                [nn.Linear(hidden_channels[-1], output_channels)]
            )
            # print(self.linear_stack)

    def forward(self, x):
        # make sure x is flattened
        for idx, layer in enumerate(self.linear_stack):
            x = layer(x)
            if idx != self.depth:
                x = torch.relu(x)
        return x

    