import pytest 
import torch 

@pytest.mark.func 
def test_tensor_equal():
    a = torch.Tensor([0,1,2])
    a_expected = torch.Tensor([0,1,2])
    assert torch.equal(a, a_expected)

@pytest.mark.func
def test_unbind():
    # torch.unbind(input, dim=0, *, out=None) -> seq
    # remove a tensor dimension
    tsr = torch.tensor([[1,2],[3,4],[5,6]])
    u, v = torch.unbind(tsr, dim=-1)
    assert u.shape == (3,)
    assert v.shape == (3,)

@pytest.mark.func 
def test_meshgrid():
    x = torch.tensor([1,2,3])
    y = torch.tensor([4,5,6])

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    assert grid_x.shape == (3,3)
    grid_x_expected = torch.tensor([[1,1,1],[2,2,2],[3,3,3]])
    assert torch.equal(grid_x, grid_x_expected)

    assert grid_y.shape == (3,3)
    grid_y_expected = torch.tensor([[4,5,6],[4,5,6],[4,5,6]])
    assert torch.equal(grid_y, grid_y_expected)

    torch.equal(torch.cat(tuple(torch.dstack([grid_x, grid_y]))), torch.cartesian_prod(x, y))

@pytest.mark.func
def test_meshgrid_xy():
    x = torch.tensor([1,2,3])
    y = torch.tensor([4,5,6])

    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    assert grid_x.shape == (3,3)
    grid_x_expected = torch.tensor([[1,2,3],[1,2,3],[1,2,3]])
    assert torch.equal(grid_x, grid_x_expected)

    assert grid_y.shape == (3,3)
    grid_y_expected = torch.tensor([[4,4,4],[5,5,5],[6,6,6]])
    assert torch.equal(grid_y, grid_y_expected)

    torch.equal(torch.cat(tuple(torch.dstack([grid_x, grid_y]))), torch.cartesian_prod(x, y))

@pytest.mark.func
def test_meshgrid_vis():
    import matplotlib.pyplot as plt 
    xs = torch.linspace(-5, 5, 100)
    ys = torch.linspace(-5, 5, 100)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    z = torch.sin(torch.sqrt(x**2 + y**2))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
    plt.show()
    assert True

@pytest.mark.func
def test_tensor_transpose():
    x = torch.tensor([[1,2,3],[4,5,6]])
    x_t = x.t()
    assert torch.equal(x_t, torch.tensor([[1,4],[2,5],[3,6]]))

@pytest.mark.func
def test_tensor_sum():
    a = torch.ones(3)
    assert torch.equal(a.sum(), torch.tensor(3.0))
    b = torch.ones(3,3)
    assert torch.equal(b.sum(), torch.tensor(9.0))

@pytest.mark.func
def test_matmul():
    a = torch.Tensor([[0.0, 1.0, 2.0]])
    b = torch.Tensor([[6.0, 7.0, 8.0]])
    c = torch.matmul(a, b.t())
    assert torch.equal(c, torch.tensor([[23.0]]))
    a_cuda = a.cuda()
    b_cuda = b.cuda()
    c_cuda = torch.matmul(a_cuda, b_cuda.t())
    print(c_cuda)
    assert torch.equal(c_cuda, torch.tensor([[23.0]]).cuda())

@pytest.mark.app
def test_batch_linear():
    x = torch.Tensor([[1,2,3],[4,5,6]])
    assert x.shape == (2,3)
    w = torch.Tensor([[7,8,9],[10,11,12]])
    assert w.shape == (2,3)
    y = torch.matmul(x, w.t())
    print(y.diag())
