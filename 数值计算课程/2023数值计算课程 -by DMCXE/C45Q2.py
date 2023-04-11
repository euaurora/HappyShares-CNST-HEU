import numpy as np
import matplotlib.pyplot as plt
from CompoundTrapezoidal import CompoundTrapezoidal as CT
from CompoundSimpson import CompoundSimpson as CS
from GaussLegendre import GaussLegendre3 as GL3
from Romberg import Romberg as R

def f1(x):
    return np.sin(2*x)/(1+x**5)

def f2(x):
    return np.sqrt(4*x)-x**2

"计算复合梯形积分值"
num1 = CT(f1,0,3,10)
num2 = CT(f2,0,2,10)
print(num1,num2)

"计算复合辛普森积分"
num1 = CS(f1,0,3,5)
num2 = CS(f2,0,2,5)
print(num1,num2)

"计算三点高斯积分"
num1 = GL3(f1,0,3)
num2 = GL3(f2,0,2)
print(num1.res(),num2.res())

"计算Romberg积分"
num1 = R(f1,0,3,0.0001)
num2 = R(f2,0,2,0.0001)
print(num1.res(),num2.res())