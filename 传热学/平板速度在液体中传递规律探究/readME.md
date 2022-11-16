# 理论背景

假设在一个以水平速度v_0匀速运动平板上分别承载无限深度的液体和有限深度（b）的液体，如图1和图2所示。

![图1 无限介质](https://user-images.githubusercontent.com/27603359/201822372-ea4a4c3a-2e0b-4215-95f9-bb59c2bd7802.png)

图1 无限介质

![图2 有限介质](https://user-images.githubusercontent.com/27603359/201822406-8849d2b1-e2a4-49f2-905e-5b8e121666fb.png)

图2 有限介质

经推导可知无限介质速度传递公式为：

$$v_x/v_0 =1-erf⁡(η)$$

其中， 
$$η=y/√4νt$$
$y$为距平板距离。

有限介质速度传递公式为：

$$\phi=1-η-\sum_{n=1}^∞{\frac{2}{n\pi} e^{-n^2 π^2 \tau} sin⁡(\eta π \eta)}$$

其中，
$$\phi=v_x/v_0$$
$$\eta=y/b$$
$$\tau=νt/b^2$$ 

# 问题

在两种情况下，当运动粘度ν=0.000001 m2⁄s，y=b=0.01 m 或 0.1m时，平板的速度何时能够传递到y处。

# C语言求解代码
```
#include<stdio.h>
#include<math.h>
#define PI 3.1415926                    //圆周率
#define nu 0.000001                     //粘度
void culc(double b);

int main()
{
  culc(0.01);
  culc(0.1);
  return 0;
}
	 
void culc(double b)
{
	double t;                           //时间
  double ni = 1, pi = 100;            //加和下限和上限
  double sum;                         //加和结果
  int m;                              //加和迭代
  long double n;                      //集中参数
  double eta;
  eta = b / 1;                      //介质厚度为1m
  for (t = 0; t < 3600; t += 0.001) {
    //有限介质
    sum = 0;
    for (m = ni; m <= pi; m++){
      sum += 2 / (m * PI) * exp(-1 * nu / (b * b) * m * m * PI * PI * t) * sin( eta * m * PI);
    }
    //无限介质
    n = b / sqrt(4 * nu * t);
	        
    if (fabs(1 – eta - sum - 1 + erfl(n)) > 0.000001){
      printf("While b = %lf m, t = %lf s\n", b, t);
      break;
    }    
  }
}
```
![image](https://user-images.githubusercontent.com/27603359/202140323-9aefd2d3-4c13-492c-a4ee-da36db36ebee.png)
![image](https://user-images.githubusercontent.com/27603359/202140446-d698bca7-e61c-4914-96ca-2bead1b8bd31.png)

## python绘制动态图
```
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
```
![图片1](https://user-images.githubusercontent.com/27603359/202140627-17023b31-8915-4cf4-bd69-53928c9718c8.gif)


