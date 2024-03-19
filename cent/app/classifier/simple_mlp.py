from module.model.network.relu_mlp import ReLUMLP
import torch.nn as nn

class SimpleMLPClassifier(nn.Module):
    def __init__(self, 
        input_dim: int = 28*28, 
        class_num: int = 10):
        super().__init__()
        self.name = "Simple MLP"
        self.input_dim = input_dim
        self.class_num = class_num
        self.network = self._build_network()

    def _build_network(self):
        return ReLUMLP(
            input_channels=self.input_dim,
            output_channels=self.class_num,
            depth = 1,
            hidden_channels=[512]
        )

    def forward(self, x):
        # batch flatten
        x = x.flatten(start_dim=1)
        return self.network(x)
