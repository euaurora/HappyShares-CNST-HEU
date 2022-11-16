import numpy as np
import time

def F(x):
    return x - np.exp(-x)
def f(x):
    return 1+x
delt = 1
x = 0.5
x0 = x
ess = 1e-10
k = 0
while delt >= ess :
    x = x - F(x)/f(x)
    delt = abs(x0-x)
    x0 = x
    k += 1
    if k == 2:
        print("sb")
        break
    print(x)

