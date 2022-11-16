import numpy as np
import matplotlib.pyplot as plt
class RK4:
    def __init__(self, F,min,max,totstep,begin):
        self.F = F
        self.min = min
        self.max = max
        self.step = (self.max-self.min)/totstep
        self.begin = begin
        self.totstep = totstep

    def slover(self):
        f = self.F
        x = 0
        y = self.begin
        yn = np.zeros(1)
        yn[0] = y
        step = self.step
        flag = 1
        for i in range(1,self.totstep+1):
            K1 = f(x,y)
            K2 = f(x+0.5*step,y+0.5*step*K1)
            K3 = f(x+0.5*step,y+0.5*step*K2)
            K4 = f(x+step,y+step*K3)
            y = y + (step/6)*(K1+2*K2+2*K3+K4)
            yn = np.append(yn,y)
            flag = flag + 1
            x = i * step
        return yn



