import numpy as np
from CubicSplineFree import CubicSplineFree as CSF

x = np.array([0,0.2,0.4,0.6,0.8,1.0])
y = np.array([0.3927,0.5672,0.6982,0.7941,0.8614,0.9053])
arr = np.c_[x,y]
csf = CSF(arr)
for i in x:
    print(csf.dnum(i))

def point3(x,y):
    h = np.abs(x[-1]-x[0])/len(x)
    df = np.array([])
    df = np.append(df,(-3*y[0]+4*y[1]-y[2])/(2*h))
    for i in range(1,len(x)-1):
        df = np.append(df,(y[i+1]-y[i-1])/(2*h))
    df = np.append(df,(3*y[-1]-4*y[-2]+y[-3])/(2*h))
    return df

def point5(x,y):
    h = np.abs(x[-1]-x[0])/len(x)
    df = np.array([])
    df = np.append(df,(y[0]-8*y[1]+8*y[2]-y[3])/(12*h))
    df = np.append(df,(y[1]-8*y[2]+8*y[3]-y[4])/(12*h))
    for i in range(2,len(x)-2):
        df = np.append(df,(y[i+2]-8*y[i+1]+8*y[i-1]-y[i-2])/(12*h))
    df = np.append(df,(y[-4]-8*y[-3]+8*y[-2]-y[-1])/(12*h))
    df = np.append(df,(y[-3]-8*y[-2]+8*y[-1]-y[-0])/(12*h))
    return df

print(point3(x,y))
print(point5(x,y))

