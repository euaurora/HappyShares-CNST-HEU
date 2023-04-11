import numpy as np

def CompoundTrapezoidal(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    xk = np.arange(a + h, b, h) #向量化操作
    s += np.sum(f(xk))
    return s * h
