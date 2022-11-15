import matplotlib.pyplot as plt
import numpy as np
import math
 
v0 = 1000           #初始速度,cm/s
nu = 0.01  #运动粘度,cm2/s
y = range(0,10)
vx = []
for t in range(1, 300):
  for i in y:
    n = i / math.sqrt(4 * nu * t)
    vx.append(v0 * (1 - math.erf(n)))
  plt.ylabel('depth(cm)')
  plt.xlabel('vx(cm/s)')
  plt.text(50,10,'t=')
  plt.text(100, 10, t)
  plt.text(150,10,'s')
  plt.barh(y, vx)
  plt.pause(0.1)
  plt.cla()
  vx.clear()
