import pytest 
import numpy as np 

from scene.bandit import Bandit, DynamicBandit
from utils.algorithm.rl.value_function import eps_greedy

import matplotlib.pyplot as plt

@pytest.mark.app
def test_k_armed_bandit():
    # create a 10-armed bandit
    bandit = Bandit(10)
    Qa = np.zeros(10)
    Na = np.zeros(10)
    n_steps = int(1e3)
    for i in range(n_steps):
        # choose an action
        A = eps_greedy(Qa)
        # get the reward
        R = bandit.step(A)
        # update the value function
        Na[A] += 1
        Qa[A] += (R - Qa[A]) / Na[A]    

    # Now the Qs is approximating the q_star_means
    assert np.argmax(Qa) == np.argmax(bandit.q_star_means)


@pytest.mark.app
def test_k_armed_dynamic_bandit():
    # create a 10-armed bandit
    bandit = DynamicBandit(10)
    Qad = np.zeros(10)
    Qa = np.zeros(10)
    Na = np.zeros(10)
    n_steps = int(1e3)
    lr = 0.02

    errd = [] # the error queue of recent 100 steps
    err = []

    for i in range(n_steps):
        # choose an action
        A = eps_greedy(Qa)
        Ad = eps_greedy(Qad)
        # get the reward
        R = bandit.step(A, i)
        Rd = bandit.step(Ad, i)
        # update the value function
        Na[A] += 1
        errd.append(Rd - Qad[A])
        err.append(R - Qa[A])
        if len(errd) > 100:
            errd.pop(0)
            err.pop(0)

        Qad[A] += (Rd - Qad[A]) * lr  
        Qa[A] += (R - Qa[A]) * 1 / Na[A]

    # The error become small enough
    assert abs(np.array(err).mean()) > 0.1 # approx 0.06
    assert abs(np.array(errd).mean()) < 0.1 # approx 0.14
