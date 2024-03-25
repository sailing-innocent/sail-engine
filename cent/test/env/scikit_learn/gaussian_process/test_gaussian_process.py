import pytest 

from module.dataset.basic.xsinx_dataset import XSNIXDataset1D, XSNIXDataset1DConfig
from app.visualizer.function.plot import FunctionPlotVisualizer
from app.visualizer.base import DummyVisualizerConfig, DummyVisualizeResult
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import matplotlib.pyplot as plt 

@pytest.mark.app
def test_gaussian_process():
    xsinx_config = XSNIXDataset1DConfig()
    xsinx_config.N_samples = 1000
    xsinx_dataset = XSNIXDataset1D(xsinx_config)
    X_train = xsinx_dataset.train_split[:, 0].reshape(-1, 1)
    y_train = xsinx_dataset.train_split[:, 1].reshape(-1, 1)
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)

    X = xsinx_dataset.x.reshape(-1, 1)
    y = xsinx_dataset.y.reshape(-1, 1)
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.show()

    # vis = FunctionPlotVisualizer()
    # vis.visualize(xsinx_dataset)

