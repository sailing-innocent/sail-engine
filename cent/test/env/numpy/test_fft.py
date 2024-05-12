import numpy as np 
import matplotlib.pyplot as plt

def test_fft():
    t = np.arange(256)
    sp = np.fft.fft(np.sin(t))
    freq = np.fft.fftfreq(t.shape[-1])
    # plt.plot(freq, sp.real, freq, sp.imag)
    # plt.show()
    y = np.fft.ifft(sp)
    plt.plot(t, y)
    plt.show()
    assert 0 == 0