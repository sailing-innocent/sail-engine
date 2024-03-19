# Test Visualize Gaussian Mixture

import pytest 

import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import linalg

import itertools
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_gaussians(means, covariances, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_alpha(0.5)
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)
    
    plt.xlim(-6.0, 6.0)
    plt.ylim(-6.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

@pytest.mark.app
def test_vis_gmm():
    means = np.array([[0.5, 0.5], [-0.5, -0.5]])
    covariances = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, -1.0], [-1.0, 1.0]]])

    plot_gaussians(means, covariances, "Gaussian Mixture")
    plt.show()