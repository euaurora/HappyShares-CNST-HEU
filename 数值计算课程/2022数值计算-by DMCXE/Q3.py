import numpy as np
import matplotlib.pyplot as plt
from Romberg import Romberg
from GaussLegendre import GaussLegendre3
from NewtonIter import NewtonIter
ess = 1e-4

def visualize(F, start, end, step):
    x = np.linspace(start, end, step)
    y = np.zeros(1)
    for i in x:
        y = np.append(y,F(i))
    y = y[1:]
    plt.figure()
    plt.plot(x, y)
    plt.show()

def f(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))

"Romberg格式积分值"
def FR(x):
    A = Romberg(f,0,x,ess)
    return A.res()-0.45

"Gauss-legendre格式积分值"
def FG(x):
    A = GaussLegendre3(f,0,x)
    return A.res()-0.45

"第一问：Romberg积分值"
visualize(FR,0,10,50)
print(Romberg(f,0,5,ess).Table())

"第二问：Gauss-legendre格式积分值"
visualize(FG,0,10,50)

"第三问:牛顿迭代法"
x0 = 0.5

NR = NewtonIter(FR,f,x0,0.0001)
xR = NR.Iter()
print(xR)

NG = NewtonIter(FG,f,x0,ess)
xG = NG.Iter()
print(xG)
