import pytest 
from module.model.encoder.positional_encoder import PositionalEncoder, PositionalEncoderConfig
import torch 

@pytest.mark.current 
def test_positional_encoder():
    config = PositionalEncoderConfig()
    assert config.input_dims == 3
    assert config.sampling_method == 'log'
    assert config.max_freq == 10
    assert config.N_freq == 4
    assert config.periodic_fns == ['sin', 'cos']
    encoder = PositionalEncoder(config)
    assert encoder.outdim == 24 # 3 * 4 * 2
    batch_size = 2
    x = torch.rand(batch_size, 3)
    y = encoder.encode(x)
    assert y.shape == (batch_size, 24)
    for batch in range(batch_size):
        for idx in range(config.N_freq):
            for ch in range(3):
                assert torch.abs(y[batch][idx * 2 * 3 + 0 * 3 + ch] - torch.sin(x[batch][ch] * (2.**(config.max_freq / (config.N_freq - 1) * idx)))) < 1e-3
                assert torch.abs(y[batch][idx * 2 * 3 + 1 * 3 + ch] - torch.cos(x[batch][ch] * (2.**(config.max_freq / (config.N_freq - 1) * idx)))) < 1e-3
    assert True 