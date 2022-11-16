import numpy as np
import matplotlib.pyplot as plt
from ODE import RK4

'''KOH生成速率的ODE'''
def F(x,y):
    k = 6.22*1e-19
    n1 = 2*1e3
    n2 = 2*1e3
    n3 = 3*1e3
    return k*((n1 - 0.5*y)**2)*((n2 - 0.5*y)**2)*((n3 - 0.75*y)**3)
"初值和边界条件"
time_start = 0
time_end = 0.2
totstep = 100
boundary0 = 0
"实例化"
A = RK4(F,time_start,time_end,totstep,boundary0)
koh = A.slover()
"0.2s时产率为"
print(koh[-1])
"趋势可视化"
plt.figure()
xx = np.linspace(time_start,time_end,totstep+1)
plt.plot(xx, koh)
plt.scatter(xx[-1],koh[-1])
plt.text(xx[-1]-0.05,koh[-1]-150,(xx[-1],koh[-1]))
plt.show()
