import numpy as np
import matplotlib.pyplot as plt
import FitSquares as fs
import CubicSplineFree as csf
import LinearInterpolation as li
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

"1. 最小二乘法拟合"
def SquareFit():
    G1 = np.c_[Parameter.p,Parameter.G1]
    G2 = np.c_[Parameter.p,Parameter.G2]
    G3 = np.c_[Parameter.p,Parameter.G3]

    fG1 = fs.FitSquares_polynomial(G1,3)
    fG2 = fs.FitSquares_polynomial(G2,3)
    fG3 = fs.FitSquares_polynomial(G3,3)

    #各项系数
    aG1 = fG1.phiprod()[0]
    aG2 = fG2.phiprod()[0]
    aG3 = fG3.phiprod()[0]
    print("G1系数：",aG1)
    print("G2系数：",aG2)
    print("G3系数：",aG3)

    #拟合误差
    deltaG1 = fG1.delta()
    deltaG2 = fG2.delta()
    deltaG3 = fG3.delta()
    print("G1误差：",deltaG1)
    print("G2误差：",deltaG2)
    print("G3误差：",deltaG3)

    #计算规定处近似值
    p_c = np.array([0.15,0.4,0.8])
    G1_assume = np.zeros(3)
    G2_assume = np.zeros(3)
    G3_assume = np.zeros(3)
    for i in range(3):
        G1_assume[i] = fG1.num(p_c[i])
        G2_assume[i] = fG2.num(p_c[i])
        G3_assume[i] = fG3.num(p_c[i])
    print("G1近似值：",G1_assume)
    print("G2近似值：",G2_assume)
    print("G3近似值：",G3_assume)

    #绘制拟合曲线
    x = np.linspace(0.1,1,100)
    y1 = np.zeros(100)
    y2 = np.zeros(100)
    y3 = np.zeros(100)
    for i in range(100):
        y1[i] = fG1.num(x[i])
        y2[i] = fG2.num(x[i])
        y3[i] = fG3.num(x[i])
    plt.plot(x,y1,label="G1")
    plt.plot(x,y2,label="G2")
    plt.plot(x,y3,label="G3")
    plt.scatter(Parameter.p,Parameter.G1)
    plt.scatter(Parameter.p,Parameter.G2)
    plt.scatter(Parameter.p,Parameter.G3)
    plt.title("SquareFit")
    plt.legend()
    plt.show()

"2. 分段线性、分段样条插值"
tau_0 = np.c_[Parameter.p,Parameter.tau_0]
tau = np.c_[Parameter.p,Parameter.tau]
xi = np.c_[Parameter.p,Parameter.xi]
beta = np.c_[Parameter.p,Parameter.beta]

def LinearInterpolation():
    #分段线性插值
    #由于分段线性插值概念上比较简单，但是代码实现上比较复杂，因此直接调用库函数
    p_c = np.array([0.25,0.75,0.95])
    tau_0_interp = np.interp(p_c,tau_0[:,0],tau_0[:,1])
    tau_interp = np.interp(p_c,tau[:,0],tau[:,1])
    xi_interp = np.interp(p_c,xi[:,0],xi[:,1])
    beta_interp = np.interp(p_c,beta[:,0],beta[:,1])
    print("tau_0 近似值_分段线性插值：",tau_0_interp)
    print("tau 近似值_分段线性插值：",tau_interp)
    print("xi 近似值_分段线性插值：",xi_interp)
    print("beta 近似值_分段线性插值：",beta_interp)

    #绘制插值曲线
    x = np.linspace(0.1,1,100)
    plt.plot(x,np.interp(x,tau_0[:,0],tau_0[:,1]),label="tau_0")
    plt.plot(x,np.interp(x,tau[:,0],tau[:,1]),label = "tau")
    plt.plot(x,np.interp(x,xi[:,0],xi[:,1]),label = "xi")
    plt.plot(x,np.interp(x,beta[:,0],beta[:,1]),label = "beta")
    plt.scatter(Parameter.p,Parameter.tau_0)
    plt.scatter(Parameter.p,Parameter.tau)
    plt.scatter(Parameter.p,Parameter.xi)
    plt.scatter(Parameter.p,Parameter.beta)
    plt.legend()
    plt.title("Linear Interpolation")
    plt.show()

def LinI():
    #分段线性插值,通过构建LinearInterpolation类实现
    p_c = np.array([0.25,0.75,0.95])
    tau_0_assume_LI = np.zeros(3)
    tau_assume_LI = np.zeros(3)
    xi_assume_LI = np.zeros(3)
    beta_assume_LI = np.zeros(3)
    for i in range(3):
        tau_0_assume_LI[i] = li.LinearInterpolation(p_c[i],tau_0)
        tau_assume_LI[i] = li.LinearInterpolation(p_c[i],tau)
        xi_assume_LI[i] = li.LinearInterpolation(p_c[i],xi)
        beta_assume_LI[i] = li.LinearInterpolation(p_c[i],beta)
    print("tau_0 近似值_分段线性插值：",tau_0_assume_LI)
    print("tau 近似值_分段线性插值：",tau_assume_LI)
    print("xi 近似值_分段线性插值：",xi_assume_LI)
    print("beta 近似值_分段线性插值：",beta_assume_LI)

    #绘制插值曲线
    x = np.linspace(0.1,1,100)
    y1 = np.zeros(100)
    y2 = np.zeros(100)
    y3 = np.zeros(100)
    y4 = np.zeros(100)
    for i in range(100):
        y1[i] = li.LinearInterpolation(x[i],tau_0)
        y2[i] = li.LinearInterpolation(x[i],tau)
        y3[i] = li.LinearInterpolation(x[i],xi)
        y4[i] = li.LinearInterpolation(x[i],beta)
    plt.plot(x,y1,label="tau_0")
    plt.plot(x,y2,label="tau")
    plt.plot(x,y3,label="xi")
    plt.plot(x,y4,label="beta")
    plt.scatter(Parameter.p,Parameter.tau_0)
    plt.scatter(Parameter.p,Parameter.tau)
    plt.scatter(Parameter.p,Parameter.xi)
    plt.scatter(Parameter.p,Parameter.beta)
    plt.legend()
    plt.title("Interpolation")
    plt.show()



def CubicSplineInterpolation():
    #分段样条插值
    fcbtau_0 = csf.CubicSplineFree(tau_0)
    fcbtau = csf.CubicSplineFree(tau)
    fcbxi = csf.CubicSplineFree(xi)
    fcbbeta = csf.CubicSplineFree(beta)

    #计算规定处近似值
    p_c = np.array([0.25,0.75,0.95])
    tau_0_assume_cb = np.zeros(3)
    tau_assume_cb = np.zeros(3)
    xi_assume_cb = np.zeros(3)
    beta_assume_cb = np.zeros(3)
    for i in range(3):
        tau_0_assume_cb[i] = fcbtau_0.num(p_c[i])
        tau_assume_cb[i] = fcbtau.num(p_c[i])
        xi_assume_cb[i] = fcbxi.num(p_c[i])
        beta_assume_cb[i] = fcbbeta.num(p_c[i])
    print("tau_0 近似值_分段样条插值：",tau_0_assume_cb)
    print("tau   近似值_分段样条插值：",tau_assume_cb)
    print("xi    近似值_分段样条插值：",xi_assume_cb)
    print("beta  近似值_分段样条插值：",beta_assume_cb)


    #绘制插值曲线
    x = np.linspace(0.1,1,100)
    y1 = np.zeros(100)
    y2 = np.zeros(100)
    y3 = np.zeros(100)
    y4 = np.zeros(100)
    for i in range(100):
        y1[i] = fcbtau_0.num(x[i])
        y2[i] = fcbtau.num(x[i])
        y3[i] = fcbxi.num(x[i])
        y4[i] = fcbbeta.num(x[i])
    plt.plot(x,y1,label="tau_0")
    plt.plot(x,y2,label="tau")
    plt.plot(x,y3,label="xi")
    plt.plot(x,y4,label="beta")
    plt.scatter(Parameter.p,Parameter.tau_0)
    plt.scatter(Parameter.p,Parameter.tau)
    plt.scatter(Parameter.p,Parameter.xi)
    plt.scatter(Parameter.p,Parameter.beta)
    plt.legend()
    plt.title("CubicSpline Interpolation")
    plt.show()


if __name__ == "__main__":
    SquareFit()
    CubicSplineInterpolation()
    LinearInterpolation()
    LinI()