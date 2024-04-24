import pytest 
import matplotlib.pyplot as plt 
import numpy as np 

from mpl_toolkits.axes_grid1 import make_axes_locatable

@pytest.mark.app
def test_locatable_axes():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    fix, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x, y)
    ax.set_aspect(1.)

    # create new axes on the left and top
    divider = make_axes_locatable(ax)

    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    # make some label invisible
    plt.setp(ax_histx.get_xticklabels() + ax_histy.get_yticklabels(), visible=False)

    # determine limits
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_yticks([0, 50, 100])
    ax_histy.set_xticks([0, 50, 100])

    plt.show()