import numpy as np
import time


class Lagrange:
    def __init__(self, arr1):
        self.arr1 = arr1
        self.arr1_x = arr1[:, 0]
        self.arr1_y = arr1[:, 1]
        self.lenth = len(arr1)
        self.denom = self.donodo(self.arr1_x)

    # 以下注释部分为优化前的重复代码
    def denominator(self):
         one = np.ones((self.lenth,self.lenth))
         arro = self.arr1_x*one
         arro[np.eye(self.lenth,dtype=bool)]=0
         res = np.prod(arro.T-arro, axis=1, where= (arro.T-arro) != 0, keepdims=False)
         return res

    def nominator(self,x):
        one = np.ones((self.lenth, self.lenth))
        arro_de = (x * one)[~np.eye(self.lenth,dtype=bool)].reshape(self.lenth,-1)
        arro2_de = (self.arr1_x*one)[~np.eye(self.lenth, dtype=bool)].reshape(self.lenth, -1)
        res = np.prod(arro_de - arro2_de, axis=1, keepdims=False)
        return res

    def donodo(self, x):
        one = np.ones((self.lenth, self.lenth))
        arro = (x * one).T
        arro_de = arro[~np.eye(self.lenth, dtype=bool)].reshape(self.lenth, -1)
        arro2_de = (self.arr1_x * one)[~np.eye(self.lenth, dtype=bool)].reshape(self.lenth, -1)
        res = np.prod(arro_de - arro2_de, axis=1, keepdims=False)
        return res

    def num(self, x):
        nom = self.donodo(x)
        return np.sum(self.arr1_y * nom / self.denom)


arr = np.array([[0, 0], [1, 2], [3, 4]])
arr2 = np.array([[0, 0], [1, 2], [3, 4], [5, 6]])
a = Lagrange(arr).lenth
b = Lagrange(arr2)
c = b.denom
print(c)
print(b.num(1))