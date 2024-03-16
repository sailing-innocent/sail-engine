import pytest 

import logging 
# fuck PIL
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import os
import torch 
from argparse import ArgumentParser
from ..scene import Scene 
from ..scene.gaussian_model import GaussianModel 
from ..arguments import ModelParams
from ..script.render import render_set
from ..arguments import PipelineParams 

class DefaultModelParams:
    def __init__(self):
        # self.model_path = "E:/dataset/gaussian_splatting/0ef97f31-5"
        # self.source_path = "E:/dataset/360_v2/bicycle"
        self.model_path = "E:/dataset/gaussian_splatting/0f1ae49a-8"
        self.source_path = "E:/dataset/nerf_synthetic/chair"
        self.white_background = False 
        self.images = "images"
        self.eval = True 
        self.resolution = -1
        self.data_device = "cuda"

def render_sets():
    with torch.no_grad():
        # gaussians
        parser = ArgumentParser(description="Testing script parameters")
        gaussians = GaussianModel(3)
        params = DefaultModelParams()
        pipeline = PipelineParams(parser)
        iterations = 30000
        scene = Scene(params, gaussians, load_iteration=iterations, shuffle=False)
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(
            params.model_path, 
            "test", 
            scene.loaded_iter, 
            scene.getTestCameras(), 
            gaussians, pipeline, background)

@pytest.mark.current 
def test_render_ply():    
    render_sets()
    assert True