import numpy as np
import matplotlib.pyplot as plt
from Lagrange import Lagrange as LA
"图像比较"
def visualize(start, end, step):
    x = np.linspace(start, end, step)
    y0 = np.zeros(1)
    y1 = np.zeros(1)
    y2 = np.zeros(1)
    for i in x:
        y0 = np.append(y0, OriginF(i))
        y1 = np.append(y1, LA1.num(i))
        y2 = np.append(y2, LA2.num(i))
    y0 = y0[1:]
    y1 = y1[1:]
    y2 = y2[1:]
    plt.figure()
    plt.scatter(X2,Y2 , c='red')
    plt.plot(x, y0,label = 'Origin F(x)')
    plt.plot(x, y1,label = 'First-Order F(x)')
    plt.plot(x, y2,label = 'Second-Order F(x)')
    plt.legend()
    plt.show()

def OriginF(x):
    return np.exp(x)

"Q1-1：对节点x0 = 0，x1 = 1进行一次差值"
X1 = np.array([0,1])    #构建X1的数组
Y1 = OriginF(X1)        #获得对应实际值的数组
AXY = np.c_[X1,Y1]      #合成N x 2点阵

LA1 = LA(AXY)
Num1 = np.array([])
for x in np.linspace(0.2,0.8,4):
    Num1 = np.append(Num1,LA1.num(x))
print(Num1)
LA1.visualize(0,2,10,True)

"Q1-2：对节点x0 = 0，x1 = 1，x2 = 0.5进行二次差值"
X2 = np.array([0,0.5,1])    #构建X1的数组
Y2 = OriginF(X2)        #获得对应实际值的数组
AXY2 = np.c_[X2,Y2]      #合成N x 2点阵

LA2 = LA(AXY2)
Num2 = np.array([])
for x in np.linspace(0.2,0.8,4):
    Num2 = np.append(Num2,LA2.num(x))
print(Num2)
LA2.visualize(0,2,10,True)

"图像比较"
visualize(0,1.5,50)