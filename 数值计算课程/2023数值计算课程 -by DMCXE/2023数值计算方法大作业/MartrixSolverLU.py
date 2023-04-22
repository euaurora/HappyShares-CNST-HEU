import numpy as np
def MartrixSolver(A, d):
    '通过LU分解求解线性方程组'
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
