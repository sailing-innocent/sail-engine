import pytest 
import matplotlib.pyplot as plt 
import numpy as np 

@pytest.mark.app
def test_histogram():
    N = 1000
    x = np.random.randn(N)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    xmax = np.max(np.fabs(x))
    binwidth = 0.25
    lim = (int(xmax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax.hist(x, bins=bins)
    # caption
    ax.set_title("Histogram")
    plt.show()
