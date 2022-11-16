import numpy as np
import matplotlib.pyplot as plt
class Romberg:
    def __init__(self, F,min,max,ess):
        self.F = F
        self.min = min
        self.max = max
        self.h = max - min
        self.ess = ess

    def x2k(self,k):
        xk = np.array([])
        for n in range(0,2 ** (k-1)):
            xk = np.append(xk,self.h*(2*n+1)/2**k)
        return xk

    def T(self,k):
        a = self.min
        b = self.max
        f = self.F
        T = np.zeros((k,k))
        T[0][0] = 0.5*self.h*(self.F(self.min)+self.F(self.max))
        for i in range(1,k):
            T[0][i] = 0.5 * T[0][i-1] \
                      + (b-a)*(2**(-i))*(np.sum(f(self.x2k(i))))

        for i in range(1,k):
            for j in range(0,k-i):
                T[i][j] = (4**i/(4**i-1))*T[i-1][j+1]\
                          - (1/(4**i-1))*T[i-1][j]
        return T

    def caculate(self):
        k = 3
        delt = 1
        while abs(delt)>self.ess:
            T = self.T(k)
            delt = T[k-1][0]-T[k-2][0]
            res = T[k-1][0]
            k = k+1
        return res,T

    def Table(self):
        return self.caculate()[1]

    def res(self):
        return self.caculate()[0]



'''
def F(x):
    return x**(3/2)

A = Romberg(F,0,2,0.0001)
print(A.res())
print(A.Table())
'''
