import numpy as np
import matplotlib.pyplot as plt
from FitSquares import FitSquares_polynomial as Fp

Arr_x = np.array([-3.5,-2.7,-1.1,-0.8,0,0.1,1.5,2.7,3.6])
Arr_y = np.array([-92.9,-85.56,-36.15,-26.52,-9.16,-8.43,-13.12,6.59,68.94])
Arr = np.c_[Arr_x,Arr_y]

'假设正交基为1,x,xn'
FP = Fp(Arr,6)
print(FP.an)
print(FP.delta())
FP.visualize(-3.6,3.7,100,False)

delta = np.array([])
plt.figure(figsize=(10, 8))
plt.scatter(Arr_x,Arr_y,c="red",zorder=0)
for i in range(1,11):
    FPi = Fp(Arr,i)
    delta = np.append(delta,FPi.delta())
    print("Length of Base Func=",i,"an=",FPi.an)
    x = np.linspace(-3.6, 3.7, 100)
    y = np.array([])
    for x0 in x:
        y = np.append(y,FPi.num(x0))
    plt.plot(x,y,label="Length of Base Func"+str(i))
    plt.legend()
plt.show()

plt.figure()
plt.plot(range(1,11),np.log10(delta),marker = "o")
plt.xlabel("Length of Base Func")
plt.ylabel("log10(DELTA)")
plt.show()
