import pytest 

import numpy as np 
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import mixture

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

@pytest.mark.current 
def test_plot_gmm():
    n_samples = 500
    np.random.seed(0)
    C = np.array([[0.0, -0.1],[1.7,0.4]])

    # Generate the Data with 2 Components
    X = np.r_[
        np.dot(np.random.randn(n_samples, 2), C),
        0.7 * np.random.randn(n_samples, 2) + np.array([-6,3])
    ]
    # 1000 samples with two blocks, one centered at (0.85,0.15) and the other (-5.5,3.5)

    gmm = mixture.GaussianMixture(n_components=5, covariance_type="full").fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")

    dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full").fit(X)
    plot_results(
        X,
        dpgmm.predict(X),
        dpgmm.means_,
        dpgmm.covariances_,
        1,
        "Bayesian Gaussian Mixture with a Dirichlet process prior",
    )

    plt.show()