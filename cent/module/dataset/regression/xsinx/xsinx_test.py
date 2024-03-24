import pytest 
from .dataset import XSNIXDataset1D, XSNIXDataset1DConfig
from mission.config.env import get_env_config

import numpy as np 

@pytest.mark.current 
def test_xsinx_default():
    env_config = get_env_config()
    N = 100
    xsinx_config = XSNIXDataset1DConfig(env_config)
    xsinx_config.sample_size = N
    xsinx_dataset = XSNIXDataset1D(xsinx_config)
    assert len(xsinx_dataset) == N
    feat, label = xsinx_dataset[0]
    assert feat.shape == (1,)
    assert label.shape == (1,)
    features = xsinx_dataset.features()
    labels = xsinx_dataset.labels()
    assert features.shape == (N,1)
    assert labels.shape == (N,1)
    xsinx_dataset.visualize()
