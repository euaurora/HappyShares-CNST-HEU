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
