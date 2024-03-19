import pytest 

import torch 
import torchvision
from torch.utils.cpp_extension import load

simple_ray_marcher = load(
    "simple_ray_marcher", [
        "lib/torch_ext/ray_tracer/simple_ray_marcher/marcher_wrapper.cpp",
        "lib/torch_ext/ray_tracer/simple_ray_marcher/marcher_wrapper.cu",
        "lib/torch_ext/ray_tracer/simple_ray_marcher/marcher.cu"
    ],
    verbose=False,
    extra_include_paths=["lib/"]
)

@pytest.mark.current 
def test_simple_marcher():
    width = 256
    height = 256
    flag, img = simple_ray_marcher.render(height, width)
    img = img.cpu()
    assert flag == 1
    torchvision.utils.save_image(img, "output/test_simple_marcher.png")