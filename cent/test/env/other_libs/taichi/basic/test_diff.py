import pytest 
import taichi as ti 

@pytest.mark.func
def test_grad():
    ti.init()
    
    N = 16
    x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def func():
        for i in x:
            loss[None] += x[i] ** 2

    for i in range(N):
        x[i] = i

    # zero_grad
    loss.grad[None] = 1

    func() # forward
    func.grad() # backward

    for i in range(N):
        assert x.grad[i] == i * 2

@pytest.mark.app 
def test_differential():
    ti.init()
    real = ti.f32
    N = 2
    x = ti.field(dtype=real, shape=(N), needs_grad=True)
    y = ti.field(dtype=real, shape=(), needs_grad=True)


    # $y = \sum_{i=1}^M{(x_i-1)^2}$
    @ti.kernel
    def compute_y():
        for i in range(N):
            y[None] += - (x[i] - 1)*(x[i] - 1)

    max_steps = 30
    x[0] = 2.0
    x[1] = 2.0
    eps = 0.1
    for step in range(max_steps):
        with ti.ad.Tape(y):
            compute_y()
            # compute_y

        message = 'dy/dx = ['
        for i in range(N):
            message = message + str(x.grad[i]) + ','

        message += "] when x = ["
        for i in range(N):
            message = message + str(x[i]) + ','

        message += "]"
        print(message)
        for i in range(N):
            x[i] += eps * x.grad[i]

    print("The Optimized Result is: ")
    for i in range(N):
        print(str(x[i]) + ',')

    # the optimization result is near {1, 1}

    assert True