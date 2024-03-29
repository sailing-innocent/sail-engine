import pytest
import numpy as np 

import matplotlib.pyplot as plt


@pytest.mark.app
def test_normal():
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, 1000) # generate 1000 instances on this normal distribution
    assert abs(mu - np.mean(s)) < 0.01
    assert abs(sigma - np.std(s, ddof=1)) < 0.5
    count, bins, ignored = plt.hist(s, 30, density=True) # draw 30 histogram
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
    plt.show()