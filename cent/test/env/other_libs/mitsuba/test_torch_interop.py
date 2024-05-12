import drjit as dr 
import mitsuba as mi 
mi.set_variant('cuda_ad_rgb')

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import matplotlib.pyplot as plt

import pytest 

@pytest.mark.func
def test_torch_drjit():
    assert True