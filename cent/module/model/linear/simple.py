import torch 

def linreg(X, w, b):
    return torch.matmul(X, w) + b

