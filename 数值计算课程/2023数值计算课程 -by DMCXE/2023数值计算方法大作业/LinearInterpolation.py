import numpy as np

def LinearInterpolation(x,arr1):
    '线性插值'
    n = len(arr1)
    for i in range(0, n):
        if arr1[i, 0] == x:
            return arr1[i, 1]
        elif arr1[i, 0] > x:
            return arr1[i - 1, 1] + (x - arr1[i - 1, 0]) * (arr1[i, 1] - arr1[i - 1, 1]) / (arr1[i, 0] - arr1[i - 1, 0])

