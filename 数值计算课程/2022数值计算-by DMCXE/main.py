from Lagrange import Lagrange as la
from Newton import Newton as Ne
from CubicSplineFree import CubicSplineFree as CS
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[1960,180671],[1970,205052],[1980,227225],[1990,249623],[2000,282162],[2010,309327],[2020,329484]])

#Lagrange插值法实例化
Lag = la(arr)
#牛顿插值法实例化
New = Ne(arr)
#三次样条插值实例化
Csf = CS(arr)

#可视化
Lag.visualize(1950,2030,1000,text=True)
New.visualize(1950,2030,1000,text=True)
Csf.visualize(1950,2030,1000,text=True)

#Lagrange部分
po0_L = Lag.num(1950)
eff0_L = abs(Lag.num(1950)-151326)/151326
po1_L = Lag.num(2005)
eff1_L = abs(Lag.num(2005)-295516)/295516
po2_L = Lag.num(2030)
print(po0_L,po1_L,po2_L,eff0_L,eff1_L)
#牛顿部分
po0_N = New.num(1950)
eff0_N = abs(New.num(1950)-151326)/151326
po1_N= New.num(2005)
eff1_N = abs(New.num(2005)-295516)/295516
po2_N = New.num(2030)
print(po0_N,po1_N,po2_N,eff0_N,eff1_N)
#三次样条部分
po0_C = Csf.num(1950)
eff0_C = abs(Csf.num(1950)-151326)/151326
po1_C = Csf.num(2005)
eff1_C = abs(Csf.num(2005)-295516)/295516
po2_C = Csf.num(2030)
print(po0_C,po1_C,po2_C,eff0_C,eff1_C)

print(New.f()[1])


'''
N = 8
arr_y = 100*np.sin(np.random.random(N))
arr_x = np.linspace(0,100,N)
arr = np.c_[arr_x,arr_y]
'''