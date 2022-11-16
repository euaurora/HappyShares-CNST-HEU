from PipeArrange import Pip_arrangement
import numpy as np
import time

s1 = 3.2
s2 = s1 * np.sqrt(3) / 2
s3 = 2 * s1
e = 0.1
r = 1.1
N = 1000

'''
s1 = 32 * 1e-3
s2 = s1 * np.sqrt(3) / 2
s3 = 2 * s1
e = 32 * 1e-3
r = 11  * 1e-3
N = 3227
'''
st = time.time()

#实例化与计算
pipe = Pip_arrangement(s1, s2, s3, e, r, N, 'Tri')
#pipe = Pip_arrangement(s1, s2, s3, e, r, N, 'Squar')
pipe.arrangement()

#参数
pos = pipe.Pippos
PipNum = pipe.PipeNum
R = pipe.R

stt = time.time()
print("耗时(s)：",stt-st)
print("管数：",PipNum)
print("套筒半径：",R)
print("位置：",pos)
#可视化
pipe.visualize()

