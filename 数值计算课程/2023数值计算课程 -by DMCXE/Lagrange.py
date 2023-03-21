import numpy as np
import matplotlib.pyplot as plt
class Lagrange:
    def __init__(self,arr1):
        self.arr1 = arr1
        self.arr1_x = arr1[:,0]
        self.arr1_y = arr1[:,1]
        self.lenth = len(arr1)
        self.denom = self.donodo(self.arr1_x)

    def donodo(self,x):
        one = np.ones((self.lenth, self.lenth))
        arro = (x*one).T
        arro_de = arro[~np.eye(self.lenth,dtype=bool)].reshape(self.lenth,-1)
        arro2_de = (self.arr1_x*one)[~np.eye(self.lenth, dtype=bool)].reshape(self.lenth, -1)
        res = np.prod(arro_de - arro2_de, axis=1, keepdims=False)
        return res

    def num(self,x):
        nom = self.donodo(x)
        return np.sum(self.arr1_y*nom/self.denom)

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



'''
arr = np.array([[0,0],[1,2],[3,4]])
arr2 = np.array([[0,0],[1,2],[3,4],[5,6]])
a = Lagrange(arr).lenth
b = Lagrange(arr2)
c = b.denom
b.Visualize(0,10)
print(c)
print(b.num(1))
'''