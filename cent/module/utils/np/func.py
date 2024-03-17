import numpy as np 
# discrete exponential functional
def discrete_exp_func(x_init, x_final, delay_steps=0, delay_mult=1.0, max_steps=1000_000):
    def helper(step):
        if step < 0 or (x_init == 0.0 and x_final == 0.0):
            return 0.0
        if delay_steps > 0:
            delay_rate = delay_mult + (1 - delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / delay_steps, 0.0, 1.0)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0.0, 1.0)
        log_lerp = np.exp(np.log(x_init) * (1 - t) + np.log(x_final) * t)
        return delay_rate * log_lerp

    return helper