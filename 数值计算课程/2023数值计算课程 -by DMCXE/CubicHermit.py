import numpy as np
import matplotlib.pyplot as plt
class CubicHermit:
    def __init__(self,arr1,df):
        self.arr1 = arr1
        self.arr1_x = arr1[:,0]
        self.arr1_y = arr1[:,1]
        self.df = df
        self.lenth = len(arr1)
    def zone(self,x):
        if x < np.min(self.arr1_x): zone = 0
        if x > np.max(self.arr1_x): zone = self.lenth-2
        for i in range(0,self.lenth-1):
            if x-self.arr1_x[i]>=0 and x-self.arr1_x[i+1]<=0:
                zone = i
        return zone
    def num(self,x):
        j = self.zone(x)
        a1 = (x-self.arr1_x[j+1])/(self.arr1_x[j]-self.arr1_x[j+1])
        a2 = (x-self.arr1_x[j])/(self.arr1_x[j+1]-self.arr1_x[j])
        I = (a1**2)*(1+2*a2)*self.arr1_y[j] + \
            (a2**2)*(1+2*a1)*self.arr1_y[j+1]+ \
            (a1**2)*(x-self.arr1_x[j])*self.df[j] + \
            (a2 ** 2) * (x - self.arr1_x[j+1]) * self.df[j+1]
        return I
    def visualize(self,start,end,step,text):
        x = np.linspace(start,end,step)
        y = np.zeros(1)
        for i in x:
            y = np.append(y,self.num(i))
        y = y[1:]
        plt.figure()
        plt.scatter(self.arr1_x, self.arr1_y, c='red')
        if text is True:
            for j in range(0,self.lenth):
                plt.text(self.arr1_x[j],self.arr1_y[j],(self.arr1_x[j],self.arr1_y[j]))
        plt.plot(x,y)
        plt.show()
