import numpy as np
import matplotlib.pyplot as plt
"""
三点高斯-勒让德求积公式
"""
class GaussLegendre3:
    def __init__(self, F, min, max):
        self.F = F
        self.min = min
        self.max = max

    def transaxis(self,t):
        return (self.max-self.min)*0.5*t + 0.5*(self.max+self.min)

    def res(self):
        res = (5/9) * self.F(self.transaxis(-np.sqrt(15)/5)) \
              + (8/9) * self.F(self.transaxis(0)) \
              + (5/9) * self.F(self.transaxis(np.sqrt(15)/5))
        res = 0.5*(self.max-self.min)*res
        return res



