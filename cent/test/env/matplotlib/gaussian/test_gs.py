import pytest 
import matplotlib.pyplot as plt
import numpy as np 

@pytest.mark.app
def test_gss():
    X  = np.linspace(-3, 3, 50)
    mu = [-0.5, 0.5]
    sigma = [1.0, 0.5]
    Y_sum = np.zeros_like(X)
    for i in range(len(mu)):
        Y = np.exp(-(X - mu[i]) ** 2 / (2 * sigma[i] ** 2)) / np.sqrt(2 * np.pi * sigma[i] ** 2)
        Y_sum += Y
        plt.plot(X, Y, label=f"mu={mu[i]}, sigma={sigma[i]}")
    
    plt.plot(X, Y_sum, label=f"sum of gaussians")
    plt.legend(
        loc="upper left",
        ncol=2,
        shadow=True,
        fancybox=True,
        borderaxespad=0.0,
        fontsize=8,
    )
    plt.show()

@pytest.mark.app
def test_gs():
    X  = np.linspace(-3, 3, 50)
    mu = 0.0
    sigma = 1.0
    Y1 = np.exp(-(X - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    plt.plot(X, Y1, label=f"mu={mu}, sigma={sigma}")
    plt.legend(
        loc="upper left",
        ncol=2,
        shadow=True,
        fancybox=True,
        borderaxespad=0.0,
        fontsize=8,
    )
    plt.show()