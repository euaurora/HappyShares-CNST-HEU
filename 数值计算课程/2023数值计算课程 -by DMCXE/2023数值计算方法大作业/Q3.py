import numpy as np
import matplotlib.pyplot as plt
import ODE
import plotly.graph_objects as go

'''
SG水位控制系统微分方程为：
x1' = G1 (qe - qv)
x2' = -x2/tau0 + G2 * qv/ tau0
x3' = x4/tau**2 - G3 * beta * qe / tau
x4' = -x3 - 2*xi*x4/tau + (2*xi*beta - 1)*G3 * qe
y(t) = x1 + x2 + x3
在各个功率点分别求解
'''
#表格内容
class Parameter():
    #功率水平
    p = np.array([0.1,0.2,0.3,0.5,1]) #10%,...100%
    G1 = np.array([0.0031,0.0035,0.0035,0.0035,0.0035])
    G2 = np.array([0.402,0.339,0.256,0.188,0.131])
    G3 = np.array([0.166,0.207,0.143,0.055,0.028])
    tau_0 = np.array([10,10.4,8.0,6.4,4.7])
    tau = np.array([19.7,12.5,10.3,13.3,6.6])
    xi = np.array([0.65,1.6,1.6,0.62,1.68])
    beta = np.array([-0.08,0.44,0.47,0.20,0.20])

#构建微分方程方程及方程组
class Functions():
    def __init__(self,G1,G2,G3,tau_0,tau,beta,xi,qe,qv):
        self.G1 = G1
        self.G2 = G2
        self.G3 = G3
        self.tau_0 = tau_0
        self.tau = tau
        self.beta = beta
        self.xi = xi
        self.qe = qe
        self.qv = qv
    def f1(self,t,x):
        return self.G1*(self.qe(t) - self.qv(t))
    def f2(self,t,x):
        x2 = x[1]
        return -x2/self.tau_0 + self.G2 * self.qv(t)/self.tau_0
    def f3(self,t,x):
        x4 = x[3]
        return x4/self.tau**2 - self.G3*self.beta*self.qe(t)/self.tau
    def f4(self,t,x):
        x3 = x[2]
        x4 = x[3]
        return -x3-2*self.xi*x4/self.tau+(2*self.xi*self.beta-1)*self.G3*self.qe(t)

#设置初始值
x0 = np.array([0,0,0,0])
step = 1000

def Situation1():
    '''
    情况一：
    qv(t) = 0 , 0 <= t <= 50
            | 0 , 0<= t < 5
    qe(t) = {
            | 1 , 5<= t <= 50
    '''     
    qv = lambda t: 0
    qe = lambda t: 1 if t>=5 else 0
    for i in range(0,5):
        Func = Functions(Parameter.G1[i],Parameter.G2[i],Parameter.G3[i]
                        ,Parameter.tau_0[i],Parameter.tau[i],Parameter.beta[i]
                        ,Parameter.xi[i],qe,qv)
        F = lambda t,x: np.array([Func.f1(t,x),Func.f2(t,x),Func.f3(t,x),Func.f4(t,x)])
        solver1 = ODE.RK4_for_equations(F,4,0,50,step,x0).slover()
        y = solver1[0,:] + solver1[1,:] + solver1[2,:]
        print('p = '+str(Parameter.p[i]))
        print(solver1)

        #绘图
        t = np.linspace(0,50,step)
        plt.plot(t,solver1[0,:],label='x1')
        plt.plot(t,solver1[1,:],label='x2')
        plt.plot(t,solver1[2,:],label='x3')
        plt.plot(t,solver1[3,:],label='x4')
        plt.plot(t,y,label='y')
        plt.title('Situation1,p = '+str(Parameter.p[i]))
        plt.legend()
        plt.show()
        '''
        #用plotly绘图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t,y=solver1[0,:],name='x1'))
        fig.add_trace(go.Scatter(x=t,y=solver1[1,:],name='x2'))
        fig.add_trace(go.Scatter(x=t,y=solver1[2,:],name='x3'))
        fig.add_trace(go.Scatter(x=t,y=solver1[3,:],name='x4'))
        fig.show()
        '''
def Situation2():
    '''
    情况二：
    qe(t) = 0 , 0 <= t <= 50

            | 0 , 0<= t < 5
    qv(t) = {
            | 1 , 5<= t <= 50

    '''
    qe = lambda t: 0
    qv = lambda t: 1 if t>=5 else 0
    for i in range(0,5):
        Func = Functions(Parameter.G1[i],Parameter.G2[i],Parameter.G3[i]
                        ,Parameter.tau_0[i],Parameter.tau[i],Parameter.beta[i]
                        ,Parameter.xi[i],qe,qv)
        F = lambda t,x: np.array([Func.f1(t,x),Func.f2(t,x),Func.f3(t,x),Func.f4(t,x)])
        solver1 = ODE.RK4_for_equations(F,4,0,50,step,x0).slover()
        y = solver1[0,:] + solver1[1,:] + solver1[2,:]
        print('p = '+str(Parameter.p[i]))
        print(solver1)

        #绘图
        t = np.linspace(0,50,step)
        plt.plot(t,solver1[0,:],label='x1')
        plt.plot(t,solver1[1,:],label='x2')
        plt.plot(t,solver1[2,:],label='x3')
        plt.plot(t,solver1[3,:],label='x4')
        plt.plot(t,y,label='y')
        plt.legend()
        plt.title('Situation2,p = '+str(Parameter.p[i]))
        plt.show()
def Situation3():
    '''
    情况三：
            | 0 , 0<= t < 20
    qe(t) = {
            | 1 , 20<= t <= 50

            | 0 , 0<= t < 5
    qv(t) = {
            | 1 , 5<= t <= 50

    '''
    qe = lambda t: 1 if t>=20 else 0
    qv = lambda t: 1 if t>=5 else 0
    for i in range(0,5):
        Func = Functions(Parameter.G1[i],Parameter.G2[i],Parameter.G3[i]
                        ,Parameter.tau_0[i],Parameter.tau[i],Parameter.beta[i]
                        ,Parameter.xi[i],qe,qv)
        F = lambda t,x: np.array([Func.f1(t,x),Func.f2(t,x),Func.f3(t,x),Func.f4(t,x)])
        solver1 = ODE.RK4_for_equations(F,4,0,50,step,x0).slover()
        y = solver1[0,:] + solver1[1,:] + solver1[2,:]
        print('p = '+str(Parameter.p[i]))
        print(solver1)

        #绘图
        t = np.linspace(0,50,step)
        plt.plot(t,solver1[0,:],label='x1')
        plt.plot(t,solver1[1,:],label='x2')
        plt.plot(t,solver1[2,:],label='x3')
        plt.plot(t,solver1[3,:],label='x4')
        plt.plot(t,y,label='y')
        plt.legend()
        plt.title('Situation3,p = '+str(Parameter.p[i]))
        plt.show()


if __name__ == "__main__":
    #Situation1()
    #Situation2()
    Situation3()