import numpy as np
import matplotlib.pyplot as plt
from FitSquares import FitSquares_polynomial as Fp
X = np.array([1,3,4,5,6,7,8,9])
Y = np.array([-11,-13,-11,-7,-1,7,17,29])
Arr = np.c_[X,Y]

'使用拟合基函数1,x,x2进行拟合'
FP = Fp(Arr,3)
print(FP.delta(),FP.an)
print("x=6.5处的值为=",FP.num(6.5))
FP.visualize(0,10,100,True)




