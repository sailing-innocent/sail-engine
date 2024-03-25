import pytest 

import torch 
import torchvision.utils as vutils
import numpy as np 
import torchvision.models as models 
from torchvision import datasets 

from tensorboardX import SummaryWriter
import datetime

@pytest.mark.app
def test_tensorboard():
    resnet = models.resnet18(weights=None)
    writer = SummaryWriter()
    sample_rate = 44100
    freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

    for n_iter in range(100):
        writer.add_scalars('data/scalar_group', {"xsinx": n_iter * np.sin(n_iter),
                                            "xcosx": n_iter * np.cos(n_iter),
                                            "arctanx": np.arctan(n_iter)}, n_iter)

    # writer.export_scalars_to_json("./all_scalars.json")
    assert True 