# Novel View Synthesis Problem Evaluator
# - psnr
# - ssim
# - lpips

from typing import NamedTuple
from module.utils.camera.basic import Camera
from module.utils.image.basic import Image
import torch
import numpy as np 
import os 
from loguru import logger 
import gc 

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class NVSEvalParams(NamedTuple):
    metric_types: list
    save: bool = False 
    save_dir: str = ""

class NVSEvaluator:
    def __init__(self):
        self.metric_types = ["psnr", "ssim", "lpips"]
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.metrics = {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips
        }
    
    def eval(self, dataset, model, renderer, params):
        result = {}
        for metric_type in params.metric_types:
            result[metric_type] = 0.0
    
        gt_path = params.save_dir + "/gt_" + dataset.obj_name
        os.makedirs(gt_path, exist_ok=True)
        res_path = params.save_dir + "/res_" + dataset.obj_name
        os.makedirs(res_path, exist_ok=True)
        tot_eval = len(dataset)

        for idx, (cam_info, img_info) in enumerate(dataset):
            torch.cuda.empty_cache()
            gc.collect()
            # print(img_info.data.shape)
            if (dataset.name == "nerf_blender" or dataset.name == "tank_temple"):
                camera = Camera() # info is FlipZ
                camera.from_info(cam_info)
                camera.flip() # but gaussian is required to use flip y
            elif (dataset.name == "mip360"):
                camera = Camera("FlipY")
                camera.from_info(cam_info)
            else:
                raise NotImplementedError()

            pred = renderer.render(camera, model)["render"].detach().cpu()
            gt = torch.from_numpy(img_info.data.transpose(2, 0, 1)).float()
            
            if (params.save):
                img = Image()
                img.load_from_info(img_info)
                img.save(os.path.join(gt_path, "{}.png".format(idx)))
                img_np = pred.numpy().transpose(1, 2, 0).clip(0, 1)
                img.load_from_data(img_np)
                img.save(os.path.join(res_path, "{}.png".format(idx)))

            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            gc.collect()
            for metric_type in params.metric_types:
                value = self.metrics[metric_type](pred, gt)
                result[metric_type] += value.item()
                logger.info(f"Done {idx}/{tot_eval} iter evaluation with {metric_type}:{value.item()}")

        for metric_type in params.metric_types:
            result[metric_type] = result[metric_type] / tot_eval

        return result