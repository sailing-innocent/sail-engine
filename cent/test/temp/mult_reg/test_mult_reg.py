# multimodel regression
import torch
def test_mult_reg():
    N_batch = 10
    N_in_dim = 3
    N_out_dim = 1
    N_sample = 4
    W = torch.arange(N_batch * N_in_dim).reshape(N_batch, N_in_dim, N_out_dim).float()
    X = torch.randn(N_batch, N_sample, N_in_dim)
    # repeat N samples
    Y = torch.matmul(X, W)
    print(Y.shape)
    assert Y.shape == (N_batch, N_sample, N_out_dim)
    # (X^T X)^-1 X^T Y
    XTX = torch.matmul(X.transpose(1, 2), X)
    # print(XTX)
    XTX_inv = XTX.inverse()
    print(XTX_inv.shape)
    XTY = torch.matmul(X.transpose(1, 2), Y)
    print(XTY.shape)
    W_next = torch.matmul(XTX_inv, XTY)
    print(W_next)
    print(W_next.shape)
    # print(B)