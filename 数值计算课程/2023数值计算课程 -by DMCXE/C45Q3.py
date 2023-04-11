import numpy as np
import matplotlib.pyplot as plt
from CubicSplineFree import CubicSplineFree as CSF

def True_D(x):
    return -70+7*x+70*np.exp(-x/10)
def True_dD(x):
    return 7-7*np.exp(-x/10)

t = np.array([8,9,10,11,12])
D = np.array([17.453,21.460,25.752,30.301,35.084])
arr = np.c_[t,D]

csf = CSF(arr)
print(csf.dnum(10))
print(True_dD(10))

x = np.linspace(7,13,100)
y1 = np.array([])
y3 = np.array([])
y2 = True_dD(x)
y4 = True_D(x)
for i in x:
    y1 = np.append(y1,csf.dnum(i))
    y3 = np.append(y3,csf.num(i))
plt.figure()
plt.scatter(t,D)
plt.plot(x,y3,lw = 2,label='CubicSplineFree-D(x)')
plt.plot(x,y4,lw = 2,linestyle='--',label='True-D(x)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,y1,lw = 2,label='CubicSplineFree-D(x)')
plt.plot(x,y2,lw = 2,linestyle='--',label='True-D(x)')
plt.legend()
plt.show()