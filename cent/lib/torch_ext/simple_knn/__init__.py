import torch 
from torch.utils.cpp_extension import load 

_C = load(
    "distCUDA2", [
        "lib/reimpl/torch_ext/simple_knn/ext.cpp",
        "lib/reimpl/torch_ext/simple_knn/simple_knn.cu",
        "lib/reimpl/torch_ext/simple_knn/spatial.cu"
    ],
    verbose=False
)

def distCUDA2(points: torch.tensor):
    return _C.distCUDA2(points)

