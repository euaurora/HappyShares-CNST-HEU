import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import time

def Lmatrix_withBound(Nx,Ny,bound,dx,dy):
    '''
    Nx: Nx=Ny,差分网格横/纵方向上的节点数
    bound: nx2数组，给出了不同边界位置的坐标
    '''
    #对边界进行变换
    bou = (np.array([num[0]+Nx*num[1] for num in bound]))

    N_grid = Nx*Ny

    e00 = np.ones(N_grid)
    e00 /= dy**2
    e00[bou] = 0
    e00 = np.roll(e00, -(Nx ** 2 - Nx))

    e000 = np.ones(N_grid)
    e000 /= dy**2
    e000[bou] = 0
    e000 = np.roll(e000, (Nx ** 2 - Nx))

    e0 = np.ones(N_grid)
    e0 /= dy**2
    e0[bou] = 0
    e0 = np.roll(e0, -Nx)

    e6 = np.ones(N_grid)
    e6 /= dy ** 2
    e6[bou] = 0
    e6 = np.roll(e6, Nx)

    e1 = np.zeros(N_grid)
    e1[np.array(range(Nx-1,N_grid,Nx))] = 1
    e1 /= dx ** 2
    e1[bou] = 0
    e1 = np.roll(e1,-Nx+1)

    e5 = np.zeros(N_grid)
    e5[np.array(range(Nx - 1, N_grid, Nx))] = 1
    e5 /= dx ** 2
    e5[bou] = 0
    e5 = np.roll(e5, Nx )

    e2 = np.ones(N_grid)
    e2[np.array(range(Nx, N_grid, Nx))]=0
    e2 /= dx ** 2
    e2[bou] = 0
    e2 = np.roll(e2,-1)

    e4 = np.ones(N_grid)
    e4[np.array(range(Nx-1, N_grid, Nx))] = 0
    e4 /= dx ** 2
    e4[bou] = 0
    e4 = np.roll(e4, 1)

    e3 = (-2*np.ones(N_grid)/dx**2)+(-2*np.ones(N_grid)/dy**2)
    #e3 /= dx ** 2
    e3[bou] = 1

    diags = np.array([-(Nx ** 2 - Nx),-Nx,-Nx+1, -1 , 0 , 1 , Nx-1,Nx,(Nx ** 2 - Nx)])
    vals = np.vstack((e00 ,e0,  e1,  e2 , e3 ,e4, e5,  e6,e000))
    #diags = np.array([ -Nx, -Nx + 1, -1, 0, 1, Nx - 1, Nx])
    #vals = np.vstack((e0,  e1,  e2 , e3 ,e4, e5,  e6))
    mtx = sp.spdiags(vals,diags,N_grid,N_grid)
    mtx = sp.lil_matrix(mtx)
    mtx = sp.csr_matrix(mtx)
    return mtx


def bound_condition(T_left,T_right,T_up,T_down,Nx,Ny,dx,dy):
    T_right = T_right
    T_left = T_left
    T_up = T_up
    T_down = T_down
    RHS = np.zeros(Nx * Ny)
    "配置边界网格"
    bound_right = (Ny - 1) * np.ones((Nx, 2))
    bound_right[:][:, 0] = np.arange(0, Nx)
    bound_liner_right = np.array([num[0] + Nx * num[1] for num in bound_right]).astype(int)

    bound_left = np.zeros((Nx, 2))
    bound_left[:][:, 0] = np.arange(0, Nx)
    bound_liner_left = np.array([num[0] + Nx * num[1] for num in bound_left]).astype(int)

    bound_up = np.zeros((Ny, 2))
    bound_up[:][:, 1] = np.arange(0, Ny)
    bound_liner_up = np.array([num[0] + Nx * num[1] for num in bound_up]).astype(int)

    bound_down = (Nx - 1) * np.ones((Ny, 2))
    bound_down[:][:, 1] = np.arange(0, Ny)
    bound_liner_down = np.array([num[0] + Nx * num[1] for num in bound_down]).astype(int)

    bound = np.vstack((bound_left, bound_right, bound_up, bound_down)).astype(int)
    RHS[bound_liner_left] = T_left
    RHS[bound_liner_right] = T_right

    RHS[bound_liner_up] = T_up
    RHS[bound_liner_down] = T_down
    Lmatx = Lmatrix_withBound(Nx,Ny, bound, dx,dy)
    return Lmatx,RHS

def main():
    Lx = 0.6
    Ly = 0.4
    Nx = 256
    Ny = 256

    dx = Lx/Nx
    dy = Ly/Ny

    "配置边界温度"
    T_right = 100
    T_left = 100
    T_up = 500
    T_down = 100

    "配置系数矩阵A与向量b"
    Get_bound = bound_condition(T_left,T_right,T_up,T_down,Nx,Ny,dx,dy)
    Lmatx = Get_bound[0]
    RHS = Get_bound[1]
    T_grid = spsolve(Lmatx,RHS)

    T_martix = T_grid.reshape(Nx, Nx).T  # 将矩阵

    return T_martix


if __name__ == "__main__":
    Lx = 0.6
    Ly = 0.4
    Nx = 256
    Ny = 256
    st = time.time()
    T_martix = main()
    end = time.time() - st
    print(end)

    fig = plt.figure(figsize=(5, 4), dpi=80)
    plt.cla()
    Tp = plt.matshow(T_martix)
    plt.colorbar(Tp)

    plt.show()


