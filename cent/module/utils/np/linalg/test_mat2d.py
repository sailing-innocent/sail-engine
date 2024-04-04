import pytest 

from .mat2d import scale_to_S, theta_to_R
from .mat2d import sym2d_to_scale_theta, scale_theta_to_sym2d
import numpy as np 


@pytest.mark.current 
def test_scale_theta_sym2d():
    s1 = 3
    s2 = 2
    theta = np.pi / 5
    sym2d = scale_theta_to_sym2d(s1, s2, theta)
    print(sym2d)
    s1_, s2_, theta_ = sym2d_to_scale_theta(sym2d)

    assert s1 == pytest.approx(s1_)
    assert s2 == pytest.approx(s2_)
    assert theta == pytest.approx(theta_)

    s1 = 3
    s2 = 2
    theta = 2 * np.pi / 5
    sym2d = scale_theta_to_sym2d(s1, s2, theta)
    print(sym2d)
    s1_, s2_, theta_ = sym2d_to_scale_theta(sym2d)

    assert s1 == pytest.approx(s1_)
    assert s2 == pytest.approx(s2_)
    assert theta == pytest.approx(theta_)

@pytest.mark.func
def test_thetaq_to_R():
    theta = np.pi / 4
    R = theta_to_R(theta)
    assert R[0][0] == np.cos(theta)
    assert R[0][1] == -np.sin(theta)
    assert R[1][0] == np.sin(theta)
    assert R[1][1] == np.cos(theta)

    thetas = np.array([np.pi / 4, np.pi / 2])
    Rs = theta_to_R(thetas)
    assert Rs[0][0][0] == np.cos(thetas[0])
    assert Rs[0][0][1] == -np.sin(thetas[0])
    assert Rs[0][1][0] == np.sin(thetas[0])
    assert Rs[0][1][1] == np.cos(thetas[0])
    assert Rs[1][0][0] == np.cos(thetas[1])
    assert Rs[1][0][1] == -np.sin(thetas[1])
    assert Rs[1][1][0] == np.sin(thetas[1])
    assert Rs[1][1][1] == np.cos(thetas[1])


@pytest.mark.func
def test_scale_to_S():
    s1 = 2
    s2 = 3
    S = scale_to_S(s1, s2)

    assert S[0][0] == 2
    assert S[1][1] == 3
    assert S[0][1] == 0
    assert S[1][0] == 0

    s1 = np.array([2, 3])
    s2 = np.array([4, 5])

    S = scale_to_S(s1, s2)

    assert S[0][0][0] == 2
    assert S[0][1][0] == 0
    assert S[0][0][1] == 0
    assert S[0][1][1] == 4

    assert S[1][0][0] == 3
    assert S[1][1][0] == 0
    assert S[1][0][1] == 0
    assert S[1][1][1] == 5

