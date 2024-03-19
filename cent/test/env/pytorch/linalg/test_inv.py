import pytest 
import torch 

@pytest.mark.func
def test_torch_inv():
    A = torch.tensor([[1, 2], [3, 4]]).float()
    A_inv = torch.linalg.inv(A)
    denom = torch.tensor(1 * 4 - 2 * 3)
    assert torch.isclose(A_inv[0][0], (4 / denom))
    assert torch.isclose(A_inv[0][1], (-2 / denom))
    assert torch.isclose(A_inv[1][0], (-3 / denom))
    assert torch.isclose(A_inv[1][1], (1 / denom))


@pytest.mark.current 
def test_batch_inv():
    A = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]).float()
    A_inv = torch.linalg.inv(A)
    denom = torch.tensor(1 * 4 - 2 * 3)
    assert torch.isclose(A_inv[0][0][0], (4 / denom))
    assert torch.isclose(A_inv[0][0][1], (-2 / denom))
    assert torch.isclose(A_inv[0][1][0], (-3 / denom))
    assert torch.isclose(A_inv[0][1][1], (1 / denom))
    assert torch.isclose(A_inv[1][0][0], (4 / denom))
    assert torch.isclose(A_inv[1][0][1], (-2 / denom))
    assert torch.isclose(A_inv[1][1][0], (-3 / denom))
    assert torch.isclose(A_inv[1][1][1], (1 / denom))