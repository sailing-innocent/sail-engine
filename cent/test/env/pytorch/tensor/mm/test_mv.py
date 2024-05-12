import pytest 
import torch 

@pytest.mark.app
def test_mv():
    m1 = torch.tensor([[1,2],[3,4]])
    Nv = torch.tensor([[5,6],[7,8], [9,10]])
    Nr = Nv @ m1
    # print(Nr)
    # assert torch.all(Nr == torch.tensor([[19, 22], [43, 50], [67, 78]]))
    m2 = torch.tensor([1,2])
    Nr2 = Nv @ m2 
    print(Nr2)


@pytest.mark.app
def test_m3v():
    N3v = torch.tensor([[[1,2], [3,4]], [[5,6],[7,8]],[[9,10],[11,12]]])
    m1 = torch.tensor([1,2])
    N3r = N3v @ m1
    assert N3r.shape == torch.Size([3, 2])

    N3v2 = torch.arange(24).reshape(4, 3, 2)
    print(N3v2)
    m2 = torch.tensor([[1,2],[3, 4], [5, 6]]) # 3, 2
    N3r2 = N3v2 @ m2
    assert N3r2.shape == torch.Size([4, 2, 2])
    print(N3r2)

@pytest.mark.current
def test_trans():
    A1 = torch.tensor([[4, 5, 6, 7],[8, 9, 1, 2]]).float()
    A3 = torch.tensor([0, 1, 2, 3]).float()
    B = torch.tensor([[[4],[5]],[[6],[7]],[[8],[9]]]).float() # 2d result
    assert B.shape == torch.Size([3, 2, 1])
    # print(B*A)
    # print(A*B)
    C = A3 * B - A1 
    assert C.shape == torch.Size([3, 2, 4])
    print(C)
    # X = torch.tensor([1,2,3,4]) # 3D result

    # Y = C @ X
    # assert Y.shape == torch.Size([3, 2])
    # print(Y)
    # D = C.reshape(-1, 4)
    # assert D.shape == torch.Size([6, 4])
    # print(D)
    X = (C.transpose(1, 2) @ C).inverse() @ C.transpose(1, 2) @ B
    print(X)