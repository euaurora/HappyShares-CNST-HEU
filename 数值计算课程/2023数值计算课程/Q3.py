import numpy as np
import matplotlib.pyplot as plt
from Lagrange import Lagrange
from CubicSplineFree import CubicSplineFree
from CubicHermit import  CubicHermit
from scipy.interpolate import CubicSpline
from scipy import interpolate as spip

"原函数"
def OriginF(x):
    return 1/(1+x**2)
"原函数的导数"
def DOriginF(x):
    return -2*x/((1+x**2)**2)
X = np.linspace(-5,5,6)
Y = OriginF(X)
AXY = np.c_[X,Y]
dY = DOriginF(X)
x0 = np.linspace(-5,5,21)

"拉格朗日差值部分"
LA = Lagrange(AXY)
Num_LA = np.array([])
for x in x0:
    Num_LA = np.append(Num_LA,LA.num(x))
print("拉格朗日=",Num_LA)
LA.visualize(-6,6,100,True)

"自由三次样条部分"
CSF = CubicSplineFree(AXY)
Num_CSF = np.array([])
for x in x0:
    Num_CSF = np.append(Num_CSF,CSF.num(x))
print("自由三次样条=",Num_CSF)
CSF.visualize(-6,6,100,True)

"带导数的分段三次Hermit差值"
CH = CubicHermit(AXY,dY)
Num_CH = np.array([])
for x in x0:
    Num_CH = np.append(Num_CH,CH.num(x))
print("分段三次Hermite=",Num_CH)
CH.visualize(-6,6,100,True)

"Scipy自带函数"
"分段线性插值"
Line = spip.interp1d(X,Y,kind='linear')
print("分段线性=",Line(x0))
xs = np.linspace(-5,5,100)
ys = Line(xs)
fig = plt.figure()
plt.scatter(X,Y , c='red')
plt.plot(xs,ys)
plt.show()

"绘制全部曲线"
fig = plt.figure()
xt = np.linspace(-6,6,100)
y_LA = np.array([])
y_CH = np.array([])
y_CSF = np.array([])
for x in xt:
    y_LA = np.append(y_LA,LA.num(x))
    y_CH = np.append(y_CH, CH.num(x))
    y_CSF = np.append(y_CSF, CSF.num(x))
print(y_LA)
llw = 0.8
plt.scatter(X,Y , c='red')
plt.plot(xt,y_LA,label = "Lagrange",lw = llw)
plt.plot(xt,y_CH,label = "CubicHermit",lw = llw)
plt.plot(xt,y_CSF,label = "CubicSplineFree",lw = llw)
plt.plot(xs,Line(xs),label = "Line",lw = llw)
plt.plot(xt,OriginF(xt),label = "Origin",lw = llw)
plt.legend()
plt.show()



