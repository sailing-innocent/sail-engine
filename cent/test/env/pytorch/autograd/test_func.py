import pytest 
import torch 

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

# this is the psedo-exp, showing that you can modify gradient as your wish (but maybe not valid)
class dExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return 2 * grad_output * result

@pytest.mark.func
def test_exp():
    a = torch.Tensor([[0,1,2],[3,4,5]])
    a.requires_grad_()
    b = Exp.apply(a)
    gradient = torch.Tensor([[1.0,1.0,1.0], [1.0,1.0,1.0]])
    b.backward(gradient=gradient)
    assert not torch.any(a.grad - b) # b = exp(a), a.grad = b * gradient

@pytest.mark.func 
def test_dexp():
    a = torch.Tensor([[0,1,2],[3,4,5]])
    a.requires_grad_()
    c = dExp.apply(a)
    gradient = torch.Tensor([[1.0,1.0,1.0], [1.0,1.0,1.0]])
    c.backward(gradient=gradient)
    assert not torch.any(a.grad - 2 * c) # c = dexp(a), a.grad = 2 * c * gradient
