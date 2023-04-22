import numpy as np
import matplotlib.pyplot as plt
import MartrixSolverLU
'''
反应堆堆芯中135X和135I的平均浓度曲线
(r_X * Sigma_f - sigma_a_X * X) * Phi_0 * n_r + lambda_I * I - lambda_X * X = 0
r_I * Sigma_f * Phi_0 * n_r - lambda_I * I = 0

线性方程组AX = b为：
    | -lambda_X + sigma_a_X * n_r , lambda_I   | | X |   | -r_X * Sigma_f * Phi_0 * n_r |
    |                                          | |   | = |                              |
    | 0                           , -lambda_I  | | I |   | -r_I * Sigma_f * Phi_0 * n_r |

'''
# 反应堆基本参数
P0 = 3000                  # MW, 反应堆功率
D = 316                    # cm, 堆芯直径
h = 355                    # cm, 堆芯高度
r_I = 0.059                # 135I的裂变产额
r_X = 0.003                # 135Xe的裂变产额
Sigma_f = 0.3358           # cm2, 裂变截面
sigma_a_X = 3.5E-18        # cm2, 135Xe吸收截面
lambda_I = 2.9E-5          # 1/s, 135I衰变常数
lambda_X = 2.1E-5          # 1/s, 135Xe衰变常数
Eff = 3.2E-11              # MWs,裂变效率
n_r = np.arange(0.1,1+0.1,0.1) # 中子的相对密度

# 计算堆芯体积
V = 2 * np.pi * (0.5 * D)**2 * h

# 计算初始平均中子注量率
Phi_0 = P0/(Eff*V)

# 构造系数矩阵
A = lambda n_r: np.array([[-lambda_X + sigma_a_X * Phi_0 * n_r, lambda_I],
                          [0, -lambda_I]])

# 构造右端项
b = lambda n_r: np.array([-r_X * Sigma_f * Phi_0 * n_r,
                          -r_I * Sigma_f * Phi_0 * n_r])

# 求解线性方程组
X = np.zeros((len(n_r),2))
Xs = np.zeros((len(n_r),2))
for i in range(len(n_r)):
    #X[i] = np.linalg.solve(A(n_r[i]),b(n_r[i]))
    X[i] = MartrixSolverLU.MartrixSolver(A(n_r[i]),b(n_r[i]))
print("X average concentration: \n",X[:,0])
print("I average concentration: \n",X[:,1])

# 绘制曲线
plt.figure()
plt.plot(n_r,X[:,0],label='135Xe')
plt.plot(n_r,X[:,1],label='135I')
plt.xlabel('n_r')
plt.ylabel('X,Y Average concentration')
plt.title('135X & 135I Average concentration')
plt.legend()
plt.show()  
