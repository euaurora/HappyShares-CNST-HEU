import numpy as np
import matplotlib.pyplot as plt
import time
L1,L2 = 0.6,0.4
x_squard,y_squard = 60,40 #确定单边的分割数量
dx,dy = L1/x_squard,L2/y_squard
k=dx/dy
print(k)
##确定精确度区间
e = 0.000000001
#构建二维温度平面，此时为0矩阵
t = np.zeros([y_squard,x_squard])
#确定边界条件
tw1,tw2 = 60. ,20.
delta = np.linspace(0,L1,x_squard)
tw3 = tw1+tw2*np.sin(np.pi*delta/L1)
t[0]= tw3
t[1:,0]=tw1
t[y_squard-1]= tw1
t[1:,x_squard-1]=tw1
#将待求解内部区域添加初始值
t[1:y_squard-2,1:x_squard-2]=70
#构建与t完全一致待收敛判断序列
#当序列f全为1时，即每一点位置处符合精确度要求
f = np.zeros_like(t)
f[0]=1
f[1:,0]=1
f[y_squard-1]=1
f[1:,x_squard-1]=1
#通过高斯-赛德尔法进行迭代
x,y=1,1
count=0
start = time.time()
while (f==1).all() == False:
    for x in range(1,x_squard-1):
        for y in range(1,y_squard-1):
            flag=t[y,x]
            t[y,x]=(k*k*t[y-1,x]+k*k*t[y+1,x]+t[y,x-1]+t[y,x+1])/(2*k*k+2)
            #判断是否符合精确度条件
            if abs(t[y,x]-flag) < e:
                f[y,x] = 1
            else:
                f[y,x] = 0
    count=count+1
    print('\r',count,end='',flush=True)
end = time.time()
print("\r高斯-赛德尔法迭代时间：%.2f秒"%(end-start))
#偷懒方法做出图像
#plt.matshow(t,cmap=plt.cm.Reds)
#不给偷懒！
'''
X = np.arange(0,y_squard,1)
Y = t[0:,30]
plt.plot(X*0.4/40,Y,label='Numerical solution')
N = np.arange(0,L2,0.01)
M = tw1+tw2*np.sin(0.3*np.pi/L1)*(np.sinh(np.pi*(0.4-N)/L1)/np.sinh(np.pi*L2/L1))
plt.plot(N,M,'--',color = "red",label='Analytical solution')
plt.xlim(0,0.4)
plt.legend( )
'''
X = np.arange(0,x_squard,1)
Y = t[0,0:]
plt.plot(X*0.6/60,Y,label='Numerical solution')
N = np.arange(0,L1,0.001)
M = tw1+tw2*np.sin(N*np.pi/L1)*(np.sinh(np.pi*L2/L1)/np.sinh(np.pi*L2/L1))
plt.plot(N,M,'--',color = "red",label='Analytical solution')
plt.xlim(0,0.6)
plt.legend( )

plt.show()
#输出矩阵，取三位有效值
print(np.around(t,3))
