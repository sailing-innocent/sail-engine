import pytest 
from ..arguments import ModelParams, PipelineParams, OptimizationParams
from ..gaussian_renderer import network_gui
from ..script.train import training
from argparse import ArgumentParser, Namespace
import torch 

class DefaultModelParams:
    def __init__(self):
        self.sh_degree = 3
        self.model_path = "E:/logs/dummy/diff_gaussian_bonsai"
        self.source_path = "E:/dataset/360_v2/bicycle"
        self.white_background = False 
        self.images = "images"
        self.eval = True 
        self.resolution = -1
        self.data_device = "cuda"

@pytest.mark.app 
def test_train_diff_gaussian():
    # gaussians
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args("")
    mip360 = ["bicycle", "garden", "kitchen", "bonsai", "counter", "room", "stump"]
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    for item in mip360:
        params = DefaultModelParams()
        params.source_path = f"E:/datasets/nerf/360_v2/{item}"
        params.model_path = f"E:/logs/dummy/diff_gaussian_{item}"
        training(
            params, 
            opt, 
            pipeline, 
            [7000, 30000], 
            [7000, 30000], 
            [], 
            args.start_checkpoint, args.debug_from)