import pytest 

import numpy as np 

@pytest.mark.app
def test_markov_chain_static_state():
    Q = np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])
    s = np.array([0.6, 0.2, 0.2])
    for i in range(100):
        s_next = s @ Q
        s = s_next

    assert np.allclose(s, np.array([0.625, 0.3125, 0.0625]), atol=1e-2)

@pytest.mark.current
def test_mcmc_sample_beta_distribution():
    import random 
    def beta_s(w, a, b):
        return w**(a-1) * (1-w)**(b-1)
            
    def beta_mcmc(N_hops, a, b, N_samples = 1000):
        assert N_samples <= N_hops
        states = []
        cur = random.uniform(0, 1)
        for i in range(N_hops):
            states.append(cur)
            next = random.uniform(0, 1)
            if min(beta_s(next, a, b) / beta_s(cur, a, b),1) > random.uniform(0, 1):
                cur = next
        return states[-N_samples:]
    
    from matplotlib import pylab as pl
    import matplotlib.pyplot as plt
    import scipy.special as ss
    pl.rcParams['figure.figsize'] = (17.0, 4.0)

    # Actual Beta PDF.
    def beta(a, b, i):
        e1 = ss.gamma(a + b)
        e2 = ss.gamma(a)
        e3 = ss.gamma(b)
        e4 = i ** (a - 1)
        e5 = (1 - i) ** (b - 1)
        return (e1/(e2*e3)) * e4 * e5
    
    # Create a function to plot Actual Beta PDF with the Beta Sampled from MCMC Chain.
    def plot_beta(a, b):
        Ly = []
        Lx = []
        i_list = np.mgrid[0:1:100j]
        for i in i_list:
            Lx.append(i)
            Ly.append(beta(a, b, i))
        pl.plot(Lx, Ly, label="Real Distribution: a="+str(a)+", b="+str(b))
        plt.hist(beta_mcmc(100000,a,b),density=True,bins =25, 
                histtype='step',label="Simulated_MCMC: a="+str(a)+", b="+str(b))
        pl.legend()
        pl.show()

    plot_beta(0.1, 0.1)
    plot_beta(1, 1)
    plot_beta(2, 3)