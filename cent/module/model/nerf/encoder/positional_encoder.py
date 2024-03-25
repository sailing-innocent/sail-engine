import torch 
import torch.nn as nn 

from typing import NamedTuple, List

class PositionalEncoderConfig(NamedTuple):
    include_input: bool = False 
    input_dims: int = 3
    sampling_method: str = 'log'
    max_freq: int = 10
    N_freq: int = 4
    periodic_fns: List[str] = ['sin', 'cos']

class PositionalEncoder:
    def __init__(self, config: PositionalEncoderConfig):
        self.config = config
        self.encode_fns = []
        self.outdim = 0
        self.create_embedding_fn()

    def create_embedding_fn(self):
        input_dim = self.config.input_dims
        output_dim = 0
        if self.config.include_input:
            self.encode_fns.append(lambda x: x)
            output_dim += input_dim
        max_freq = self.config.max_freq
        N_freq = self.config.N_freq
        periodic_fns = self.config.periodic_fns

        if self.config.sampling_method == 'log':
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freq)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freq)

        # print(freq_bands)
        for freq in freq_bands:
            if 'sin' in periodic_fns:
                self.encode_fns.append(lambda x, freq=freq: torch.sin(x * freq))
                output_dim += input_dim
            if 'cos' in periodic_fns:
                self.encode_fns.append(lambda x, freq=freq: torch.cos(x * freq))
                output_dim += input_dim

        self.outdim = output_dim

    def encode(self, x):
        return torch.cat([fn(x) for fn in self.encode_fns], -1)

