import numpy as np
import matplotlib.pyplot as plt
from Newton import Newton

"待差值的原始数据"
AXY = np.array([[-1,-2],[1,0],[3,-6],[4,9]])
NT = Newton(AXY)
"计算x=0,2,2.5的近似值"
x0 = np.array([0,2,2.5])
y0 = np.array([])
for x in x0:
    y0 = np.append(y0,NT.num(x))
print(y0)
"生成图像"
NT.visualize(-2,5,1000,True)
"生成差值表"
Table = NT.f()[1]
print(Table)