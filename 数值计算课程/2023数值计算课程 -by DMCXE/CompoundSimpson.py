import numpy as np

def CompoundSimpson(f, a, b, n):
    h = (b - a) / n
    s = f(a) + f(b)
    xk = np.arange(a + h, b, h)
    s += 2 * np.sum(f(xk))
    xk = np.arange(0.5*h, b, h)
    s += 4 * np.sum(f(xk))
    return s * h / 6

