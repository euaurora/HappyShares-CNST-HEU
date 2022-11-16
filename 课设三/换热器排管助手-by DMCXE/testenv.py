from PipeArrange import Pip_arrangement
import numpy as np
import time

s1 = 32 * 1e-3
s2 = s1 * np.sqrt(3) / 2
s3 = 2 * s1
e = 32 * 1e-3
r = 11  * 1e-3
N = 5200

st = time.time()

#实例化与计算
#pipe = Pip_arrangement(s1, s2, s3, e, r, N, 'Tri')
pipe = Pip_arrangement(1.40*19*1e-3, s2, e, e, r, N, 'Squar')
pipe.arrangement()

#参数
pos = pipe.Pippos
PipNum = pipe.PipeNum
R = pipe.R
R_part = pipe.R_part

stt = time.time()
print("耗时(s)：",stt-st)
print("管数：",PipNum)
print("套筒半径：",R)
print("位置：",pos)
print("弯头部分总长",R_part)
#可视化
pipe.visualize()

