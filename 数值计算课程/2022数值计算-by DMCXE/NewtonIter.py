import numpy as np
from Romberg import Romberg
class NewtonIter:
    def __init__(self, F, f, x0,ess):
        self.F = F
        self.f = f
        self.x0 = x0
        self.ess = ess

    def Iter(self):
        delt = 1
        x = self.x0
        x0 = x
        count = 0
        MaxIter = 10000
        while delt >= self.ess :
            x = x - self.F(x)/self.f(x)
            if abs(x)<1:
                delt = abs(x0-x)
            else:
                delt = abs((x0-x)/x)
            x0 = x
            count += 1
            if count >= MaxIter:
                x = "超过最大迭代上限10000"
                break
        return x
'''
def F(x):
    return x*np.exp(x)-1
def f(x):
    return np.exp(x)*(1+x)
A = NewtonIter(F,f,5,0.0000001)
print(A.Iter())
'''
