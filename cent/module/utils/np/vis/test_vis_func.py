import pytest 
from module.utils.np.func import discrete_exp_func
import matplotlib.pyplot as plt

@pytest.mark.current 
def test_vis_discrete_exp_func():
    func = discrete_exp_func(1e-3, 1e-5, 100, 0.5, 1000)
    steps = list(range(1000))
    values = [func(step) for step in steps]
    plt.plot(steps, values)
    plt.show()
