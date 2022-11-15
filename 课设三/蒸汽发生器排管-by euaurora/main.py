import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from Ui_mainWindow import *
from Ui_plot import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import math
from Ui_mainWindow import *
from Ui_plot import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import math
from iapws import IAPWS97
from PIL import Image
import numpy as np
import global_var

PI = 3.1415926

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_cal.clicked.connect(self.para)
        # 设置输入格式只能为浮点数
        doubleValidator = QDoubleValidator(self)
        doubleValidator.setNotation(QDoubleValidator.StandardNotation)
        self.lineEdit.setValidator(doubleValidator)
        self.lineEdit_2.setValidator(doubleValidator)
        self.lineEdit_3.setValidator(doubleValidator)
        self.lineEdit_4.setValidator(doubleValidator)
        self.lineEdit_5.setValidator(doubleValidator)
        self.lineEdit_6.setValidator(doubleValidator)
        self.lineEdit_7.setValidator(doubleValidator)
        self.lineEdit_8.setValidator(doubleValidator)
        self.lineEdit_9.setValidator(doubleValidator)
        self.lineEdit_10.setValidator(doubleValidator)

    def para(self):
        # 检查是否为空
        if(len(self.lineEdit.text()) == 0 or len(self.lineEdit_2.text()) == 0 or len(self.lineEdit_3.text()) == 0 or           
            len(self.lineEdit_4.text()) == 0 or len(self.lineEdit_5.text()) == 0 or len(self.lineEdit_6.text()) == 0 or           
            len(self.lineEdit_7.text()) == 0 or len(self.lineEdit_8.text()) == 0 or len(self.lineEdit_9.text()) == 0 or           
            len(self.lineEdit_10.text()) == 0 or len(self.lineEdit_11.text()) == 0 or len(self.lineEdit_12.text()) == 0 or
            len(self.lineEdit_13.text()) == 0):
            # 显示提示框
            QMessageBox.warning(self, "警告", "有空值！")
        else:
            D = float(self.lineEdit.text())             # 蒸汽产量
            P_s1 = float(self.lineEdit_2.text())        # 冷却剂压力
            T_1i = float(self.lineEdit_3.text())        # SG冷却剂入口温度(℃)
            dT = float(self.lineEdit_4.text())          # SG冷却剂进出口温差(℃)
            P_s2 = float(self.lineEdit_5.text())        # 二次侧饱和压力
            T_g = float(self.lineEdit_6.text())         # 二次侧给水温度
            R_f = float(self.lineEdit_7.text())         # 传热管管壁污垢热阻
            D_d = float(self.lineEdit_8.text()) / 100   # 蒸发器排污量为蒸发量的(%)
            d_i = float(self.lineEdit_9.text()) / 1000  # 传热管内径
            d_o = float(self.lineEdit_10.text()) / 1000 # 传热管外径
            u = float(self.lineEdit_11.text())          # 冷却剂流速
            lambda_w = float(self.lineEdit_12.text())   # 传热管导热系数
            eff = float(self.lineEdit_13.text()) / 100  # 换热效率
            self.cal(D, P_s1, T_1i, dT, P_s2, T_g, R_f, D_d, d_i, d_o, u, lambda_w, eff)

    

    def cal(self, D, p_s1, t_1i, dT, p_s2, t_g, R_f, D_d, d_i, d_o, u, lamda_w, eff):
        def err(a, b, c):
            d = math.fabs(a - b)
            if (d / b < c):
                return 0
            else:
                return 1
        water1i = IAPWS97(P=p_s1, T=t_1i + 273.15)   # 蒸汽发生器入口水
        t_1o = t_1i - dT                             # 一回路蒸汽发生器出口温度
        water1o = IAPWS97(P=p_s1, T=t_1o + 273.15)   # 蒸汽发生器出口水
        t_m = (t_1o + t_1i) / 2                      # 定性温度
        water1_m = IAPWS97(P=p_s1, T=t_m)            # 定性水
        water2_s = IAPWS97(P=p_s2, x=0)              # 定义蒸汽发生器饱和水
        gas2 = IAPWS97(P=p_s2, x=1)                  # 定义新蒸汽
        water2_g = IAPWS97(P=p_s2, T=t_g + 273.15)   # 定义二回路给水
        t_s2 = water2_s.T - 273.15                   # 蒸汽发生器饱和水温度
        delta_t = dT / math.log((t_1i - t_s2) / (t_1o - t_s2))  # 平均对数温差
        nu = water1_m.nu                                # 一回路冷却剂运动粘度
        Pr = water1_m.Pr                                # 一回路冷却剂普朗特数
        lamda1 = water1_m.k                             # 一回路水的导热系数
        Re = u * d_i / nu                               # 雷诺数
        Nu = 0.023 * math.pow(Re, 0.8) * math.pow(Pr, 0.3)
        alpha_i = Nu * lamda1 / d_i                     # 管内强制对流换热系数
        Rw = (d_o / (2 * lamda_w)) * math.log(d_o / d_i)# 管壁导热热阻
        k = 2000                                        # k迭代初值
        k_new = 1000
        while (err(k, k_new, 0.05)):
            k = k_new
            q = k * delta_t
            p = p_s2 * 1e6
            alpha_o = 0.557 * pow(p, 0.15) * pow(q, 0.7)
            k_new = 1 / (d_o / (d_i * alpha_i) + Rw + 1 / alpha_o + R_f)
        h_s = water2_s.h
        h_f = water2_g.h
        h_gas = gas2.h
        r = h_gas - h_s                                 # 二回路水的汽化潜热
        Q = (D * r + (D + D_d* D) * (h_s - h_f))        # 换热量kj/s
        F = 1.09 * 1000 * Q / k / delta_t               # 换热面积
        print("总换热系数：",k)
        print("平均对数温差：", delta_t)
        print("换热面积为：", F)

        h_1i = water1i.h
        h_1o = water1o.h

        G = Q / eff / (h_1i - h_1o)                     # 一回路冷却剂流量
        rho = water1_m.rho                              # 一回路冷却剂密度
        A = math.pi * d_i * d_i / 4                     # 单根传热管的面积
        N_sum = math.ceil(G / A / rho / u)              # 管子总数
        C = math.pi * d_o                               # 换热管周长
        l_sum = F / C                                   # 换热管总长
        l_c = l_sum / N_sum                             # 平均换热管长
        print("平均传热管长(m)：", l_c)
        print("传热管总数：", N_sum)
        # 全局变量
        global_var.set_value('d_o', d_o)
        global_var.set_value('N_sum', N_sum)
        global_var.set_value('k', k)
        global_var.set_value('F', F)                       # 传热面积
        global_var.set_value('l_c', l_c)                       

        # 创建新窗口计算并展示布局图和结果数据
        self.plotWindow = plot()
        self.plotWindow.show()


# 新窗口展示布局图
class plot(QWidget):
    def __init__(self):
        super(plot, self).__init__()  # 子窗口的实例
        # self.setupUi(self)
        self.setWindowTitle("Plot")
        self.resize(1000, 900)       # 窗口大小
    def paintEvent(self, event):
        label_1 = QLabel(self)
        label_2 = QLabel(self)
        label_3 = QLabel(self)
        label_4 = QLabel(self)
        label_5 = QLabel(self)

        label_1.setGeometry(QtCore.QRect(0, 0, 1000, 20))
        label_2.setGeometry(QtCore.QRect(0, 30, 1000, 20))
        label_3.setGeometry(QtCore.QRect(0, 60, 1000, 20))
        label_4.setGeometry(QtCore.QRect(0, 90, 1000, 20))
        label_5.setGeometry(QtCore.QRect(0, 120, 1000, 20))

        painter = QPainter()
        painter.begin(self)
        # 绘制图形
        ## 获取参数
        N_sum = global_var.get_value('N_sum')               # 传热管总数
        d_o = global_var.get_value('d_o')
        k = global_var.get_value('k')                       # 传热系数
        F = global_var.get_value('F')                       # 传热面积
        d = 10                # 一根传热管等效直径
        width = 1.4 * d       # 矩形宽度，也就是截距
        # 找到最大圆半径，也就是管束直径
        i = 0
        R = math.ceil(math.sqrt(N_sum * width * width / PI)) + 1 * width
        l_c = global_var.get_value('l_c')
        R_c= R / 2 / width * 1.4 * d_o          # 平均管弯管半径
        print("==================================================================================")
        print("平均管弯头区长度：", PI * R_c)
        L = (l_c - PI * R_c) / 2    # 直管段长度（不考虑管板）
        print("直管段长度：", L)

        for x in range(2 * int(width), int(R), int(width)):
            for y in range(0, int(R), int(width)):
                if math.sqrt(x * x + y * y) < R :
                    # 画矩形
                    painter.setPen(QColor(0,0,0))
                    painter.drawRect(x+500, y+500, int(width), int(width))      # 参数1 参数2：矩形左上角坐标； 参数3：宽度； 参数4：高度

                    # 画圆
                    painter.setBrush(QColor(255,255,255))
                    painter.drawArc(int(x + 0.2 * d+500), int(y + 0.2 * d+500), int(d), int(d), 0, 360*16)

                    # 对称
                    painter.drawRect(-x+500, y+500, int(width), int(width))
                    painter.drawRect(x+500, -y+500, int(width), int(width))
                    painter.drawRect(-x+500, -y+500, int(width), int(width))
                    painter.drawArc(int(-x + 0.2 * d+500), int(y + 0.2 * d+500), int(d), int(d), 0, 360*16)
                    painter.drawArc(int(x + 0.2 * d+500), int(-y + 0.2 * d+500), int(d), int(d), 0, 360*16)
                    painter.drawArc(int(-x + 0.2 * d+500), int(-y + 0.2 * d+500), int(d), int(d), 0, 360*16)
                    i = i + 1
            if i > (N_sum / 4):
                break

        print("布置管子总数：", i * 4)

        num_lagan = i * 4 - N_sum       # 拉杆数     
        num_lagan_quard = (num_lagan - 2) / 4       
        theta = 0
        dtheta = 90 / num_lagan_quard
        # 找拉杆
        R_lagan = R / 2     # 两边中心点的管子

        # 画两边
        # painter_cir.drawArc(R_lagan * width + 0.2 * d+500, 0+500, d, d, 0, 360*16)
        # painter_cir.drawArc(-R_lagan * width + 0.2 * d+500, 0+500, d, d, 0, 360*16)
        # for x in range(int(R), 2 * int(width), -int(width)):
        #     y = R_lagan * math.sin()

        # painter_cir.end()
        painter.end()


        R = R / width * 1.4 * d_o
        print("管束直径(m)：", 2 * R)
        # 输出关键设计数据
        label_1.setText("总换热系数(W/(m2·K))=" + str(k))
        label_2.setText("传热管总数=" + str(N_sum))
        label_3.setText("传热面积(m2)=" + str(F))
        label_4.setText("管束直径(m)=" + str(2 * R))
        label_5.setText("应布置拉杆数=" + str(num_lagan))
        label_1.show()
        label_2.show()
        label_3.show()
        label_4.show()
        label_5.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    global_var._init()  # 全局变量处理器初始化
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
