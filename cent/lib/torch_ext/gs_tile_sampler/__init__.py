# modify from gaussian sampler
from typing import NamedTuple
import torch.nn as nn
import torch
from torch.utils.cpp_extension import load 

_C = load(
    "sample_gaussians", [
        "lib/torch_ext/gs_tile_sampler/sampler_wrapper.cpp",
        "lib/torch_ext/gs_tile_sampler/sampler_points.cu",
        "lib/torch_ext/gs_tile_sampler/sampler_impl.cu",
        "lib/torch_ext/gs_tile_sampler/forward.cu",
        "lib/torch_ext/gs_tile_sampler/backward.cu"
    ],
    verbose=False,
    extra_include_paths=["lib/ext"]
)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def sample_gaussians(
    means3D,
    means2D,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    sample_settings,
):
    return _SampleGaussians.apply(
        means3D,
        means2D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        sample_settings,
    )

class _SampleGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        sample_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            sample_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            sample_settings.scale_modifier,
            cov3Ds_precomp,
            sample_settings.viewmatrix,
            sample_settings.projmatrix,
            sample_settings.tanfovx,
            sample_settings.tanfovy,
            sample_settings.image_height,
            sample_settings.image_width,
            sample_settings.prefiltered,
            sample_settings.debug
        )

        # Invoke C++/CUDA sampler
        if sample_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.forward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.forward(*args)

        # Keep relevant tensors for backward
        ctx.sample_settings = sample_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp, means3D, scales, 
            rotations, cov3Ds_precomp, radii, 
            geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # print(grad_out_color[:10,:10])
        # psedo_grad = - torch.ones_like(grad_out_color, dtype=torch.float32, device="cuda")

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        sample_settings = ctx.sample_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (sample_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                sample_settings.scale_modifier, 
                cov3Ds_precomp, 
                sample_settings.viewmatrix, 
                sample_settings.projmatrix, 
                sample_settings.tanfovx, 
                sample_settings.tanfovy, 
                grad_out_color, 
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                sample_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if sample_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianSampleSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianSampler(nn.Module):
    def __init__(self, sample_settings):
        super().__init__()
        self.sample_settings = sample_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            sample_settings = self.sample_settings
            visible = _C.mark_visible(
                positions,
                sample_settings.viewmatrix,
                sample_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        sample_settings = self.sample_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA sampleization routine
        return sample_gaussians(
            means3D,
            means2D,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            sample_settings, 
        )

