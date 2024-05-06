import torch 
from torch.utils.cpp_extension import load

_C = load(
    "dummy", [
        "cent/lib/torch_ext/dummy_add/py_wrapper.cpp",
        "cent/lib/torch_ext/dummy_add/torch_wrapper.cu",
        "cent/lib/torch_ext/dummy_add/kernel.cu"
    ],
    verbose=False
)

def dummy_add(a: torch.tensor, b: torch.tensor):
    return _C.dummy_add(a, b)