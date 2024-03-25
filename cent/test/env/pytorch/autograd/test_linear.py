import torch
import pytest 

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias
        ctx.save_for_backward(input, weight, bias)
        return output # here is the output w.r.t the grad_output in backward
    
    # this function has only a single output in forward, so it gets only one gradient
    @staticmethod 
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None 

        # These needs_input_grad checks are optional 
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) # chain rule, dz/dx = dz/dy * dy/dx = dz/dy * w
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) # chain rule, dz/dw = dz/dy * dy/dw = (dz/dy)^T * x
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output # chain rule, dz/db = dz/dy * dy/db = dz/dy * 1

        return grad_input, grad_weight, grad_bias

def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)

@pytest.mark.func
def test_linear():
    a = torch.Tensor([[0,1,2],[3,4,5]])
    a.requires_grad_()
    w = torch.Tensor([[6,7,8],[9,10,11]])
    w.requires_grad_()
    b = torch.Tensor([[1.0, 1.0],[1.0, 1.0]])
    b.requires_grad_()
    c = linear(a, w, b)
    gradient = torch.Tensor([[1.0,1.0], [1.0,1.0]])
    c.backward(gradient=gradient)
    # print(a.grad) # gradient * c = 1 1 * 6 7 8  -> [15,17,19]
                  #                1 1   9 10 11   [15,17,19]
    assert not torch.any(a.grad - torch.Tensor([[15,17,19],[15,17,19]]))
    assert not torch.any(w.grad - torch.Tensor([[3,5,7],[3,5,7]]))
    assert not torch.any(b.grad - torch.Tensor([[1,1],[1,1]]))
    
from torch.autograd import gradcheck

@pytest.mark.func
def test_grad_check():
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
    assert gradcheck(linear, input, eps=1e-6, atol=1e-4)
    
import torch.nn as nn 

class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features 

        # nn.Parameter is a special kind of Tensor, that will 
        # automatically registered as Module's parameter once it's assigned

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the 
            # optional ones can be None if you want
            self.register_parameter('bias', None)

        # not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

@pytest.mark.func
def test_linear_module():
    a = torch.Tensor([[0,1,2]])
    a.requires_grad_()
    a = a.cuda()
    net = Linear(3, 2)
    net = net.cuda()
    c = net(a)
    c = c.cpu()
    assert c.shape == (1,2)
    assert True