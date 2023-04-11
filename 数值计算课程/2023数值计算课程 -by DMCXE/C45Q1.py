import numpy as np
import matplotlib.pyplot as plt
from CompoundTrapezoidal import CompoundTrapezoidal as CT
from CompoundSimpson import CompoundSimpson as CS
"根据题目内容定义三个函数f以及其导函数df"
def f1(x):
    return x**3
def df1(x):
    return 3*x**2

def f2(x):
    return np.sin(x)
def df2(x):
    return np.cos(x)

def f3(x):
    return np.exp(-x)
def df3(x):
    return -np.exp(-x)

def S(x,f,df):
    return 2*np.pi*f(x)*np.sqrt(1+df(x)**2)

S1 = lambda x: S(x,f1,df1)
S2 = lambda x: S(x,f2,df2)
S3 = lambda x: S(x,f3,df3)


"计算复合梯形积分值"
num1 = CT(S1,0,1,10)
num2 = CT(S2,0,np.pi/4,10)
num3 = CT(S3,0,1,10)
print(num1,num2,num3)

"计算复合辛普森积分"
num1 = CS(S1,0,1,5)
num2 = CS(S2,0,np.pi/4,5)
num3 = CS(S3,0,1,5)
print(num1,num2,num3)

