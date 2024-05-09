import torch 
from torch.utils.cpp_extension import load 
import os 

_C = load(
    "dummy_use", [
        "cent/lib/torch_ext/dummy_use/torch_wrapper.cu",
        "cent/lib/torch_ext/dummy_use/py_wrapper.cpp"
    ],
    verbose=False,
    extra_include_paths=[
        "internal/cu/include",
        "modules/base/include",
        "modules/core/include",
        "bin/release"
    ],
    extra_cflags=["-DSAIL_CU_API=SAIL_IMPORT"],
    extra_cuda_cflags=["-DSAIL_CU_API=SAIL_IMPORT"],
    extra_ldflags=["SailCu.lib"],
    build_directory=os.path.join(os.path.dirname(__file__), "../../../../bin/release")
)

def dummy_use(a: torch.tensor, b: torch.tensor):
    return _C.dummy_use(a, b)