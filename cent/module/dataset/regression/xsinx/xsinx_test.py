import pytest 
from .dataset import XSNIXDataset1D, XSNIXDataset1DConfig
from module.config.env import get_env_config_by_name  

import numpy as np 

@pytest.mark.current 
def test_xsinx_default():
    env_config = get_env_config_by_name("pc")
    xsinx_config = XSNIXDataset1DConfig(env_config)
    xsinx_dataset = XSNIXDataset1D(xsinx_config)
    assert len(xsinx_dataset) == 100
    feat, label = xsinx_dataset[0]
    assert feat.shape == (1,)
    assert label.shape == (1,)
    features = xsinx_dataset.features()
    labels = xsinx_dataset.labels()
    assert features.shape == (100,1)
    assert labels.shape == (100,1)
    xsinx_dataset.visualize()
