import torch 
from torch.utils.cpp_extension import load 
import os 

proj_path = os.path.join(os.path.dirname(__file__), "../../../..")
bin_path = os.path.join(proj_path, "bin/release")

_C = load(
    "dummy_use", [
        os.path.join(os.path.dirname(__file__), "torch_wrapper.cu"),
        os.path.join(os.path.dirname(__file__), "py_wrapper.cpp")
    ],
    verbose=False,
    extra_include_paths=[
        os.path.join(proj_path, "internal/cu/include"),
        os.path.join(proj_path, "modules/base/include"),
        os.path.join(proj_path, "modules/core/include"),
        os.path.join(proj_path, "bin/release")
    ],
    extra_cflags=["-DSAIL_CU_API=SAIL_IMPORT"],
    extra_cuda_cflags=["-DSAIL_CU_API=SAIL_IMPORT"],
    extra_ldflags=["SailCu.lib"],
    build_directory = bin_path 
)

def dummy_use(a: torch.tensor, b: torch.tensor):
    return _C.dummy_use(a, b)