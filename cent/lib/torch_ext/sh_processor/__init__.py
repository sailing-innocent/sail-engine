# modify from gaussian sampler
from typing import NamedTuple
import torch.nn as nn
import torch
from torch.utils.cpp_extension import load 

_C = load(
    "sample_gaussians", [
        "lib/torch_ext/sh_processor/processor_wrapper.cpp",
        "lib/torch_ext/sh_processor/processor.cu",
        "lib/torch_ext/sh_processor/processor_impl.cu",
        "lib/torch_ext/sh_processor/forward.cu",
        "lib/torch_ext/sh_processor/backward.cu"
    ],
    verbose=False,
    extra_include_paths=["lib/ext"]
)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def process_sh(
    sh,
    settings,
):
    return _SHProcessor.apply(
        sh,
        settings
    )

class _SHProcessor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sh,
        settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            sh,
            settings.sh_degree
        )

        color, geom_buffer = _C.forward(*args)
        # Keep relevant tensors for backward
        ctx.settings = settings 
        ctx.save_for_backward(
            sh, geom_buffer)
        return color

    @staticmethod
    def backward(ctx, grad_color, _):
        settings = ctx.settings
        sh, geom_buffer = ctx.saved_tensors
        grads = (
            None,
            None,
        )

        return grads

class SHProcessorSettings(NamedTuple):
    sh_degree : int

class SHProcessor(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def forward(self, shs):
        settings = self.settings
        # Invoke C++/CUDA sampleization routine
        return process_sh(
            shs,
            settings
        )
