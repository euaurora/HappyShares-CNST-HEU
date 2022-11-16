import numpy as np
import matplotlib.pyplot as plt
import time
class FitSquares_polynomial:
    def __init__(self,arr1,n):
        self.arr1 = arr1
        self.arr1_x = arr1[:,0]
        self.arr1_y = arr1[:,1]
        self.lenth = len(arr1)
        self.n = n
        self.an = self.phiprod()[0]

    def phiprod(self):
        #确定总长度
        n = self.n
        #初始化G,d向量
        G = np.array([])
        d = np.array([])
        #计算并生成G,d向量
        for i in range(0,n):
            d = np.append(d,np.sum((self.arr1_y)*(self.arr1_x**i)))
            for j in range(0,n):
                #这里的G向量是有n个元素的行向量
                G = np.append(G,np.sum((self.arr1_x**i)*(self.arr1_x**j)))
        #通过.reshape方法将G向量转为n阶方阵
        G = G.reshape(n,n)
        #通过np求逆求解，待更新轮子解法
        #an = np.dot(np.linalg.inv(G), d)
        #通过自制LU求解器
        an = self.MartrixSolver(G,d)
        return an,G,d

    def num(self,x):
        num = 0
        for i in range(0,self.n):
            num = num+(self.an[i])*(x**i)
        return num

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

    def delta(self):
        de = np.zeros(self.lenth)
        for i in range(0,self.lenth):
            de[i] = (self.num(self.arr1_x[i])-self.arr1_y[i])**2
        return np.min(de)

    #LU分解
    def MartrixSolver(self,A, d):
        n = len(A)
        U = np.zeros((n, n))
        L = np.zeros((n, n))

        for i in range(0, n):
            U[0, i] = A[0, i]
            L[i, i] = 1
            if i > 0:
                L[i, 0] = A[i, 0] / U[0, 0]
        # LU分解
        for r in range(1, n):
            for i in range(r, n):
                sum1 = 0
                sum2 = 0
                ii = i + 1
                for k in range(0, r):
                    sum1 = sum1 + L[r, k] * U[k, i]
                    if ii < n and r != n - 1:
                        sum2 = sum2 + L[ii, k] * U[k, r]
                U[r, i] = A[r, i] - sum1
                if ii < n and r != n - 1:
                    L[ii, r] = (A[ii, r] - sum2) / U[r, r]
        # 求解y
        y = np.zeros(n)
        y[0] = d[0]

        for i in range(1, n):
            sumy = 0
            for k in range(0, i):
                sumy = sumy + L[i, k] * y[k]
            y[i] = d[i] - sumy
        # 求解x
        x = np.zeros(n)
        x[n - 1] = y[n - 1] / U[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            sumx = 0
            for k in range(i + 1, n):
                sumx = sumx + U[i, k] * x[k]
            x[i] = (y[i] - sumx) / U[i, i]

        return x








'''
arr1 = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
arr2 = np.array([[1,1.629],[1.25,1.756],[1.5,1.876],[1.75,2.008],[2,2.135]])
st = time.time()
a = FitSquares_polynomial(arr2,5)
#a.visualize(0,10,100,False)
print(a.phiprod()[0])
print(a.num(1))
print(a.delta())
stt = time.time()
print(stt-st)
'''
