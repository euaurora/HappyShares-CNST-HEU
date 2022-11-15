from main import MyMain
from PyQt5.QtCore import QObject, pyqtSignal
import global_var
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from Ui_Parameters import Ui_Parameters
import json

class ParWindow(QDialog, Ui_Parameters):
    parWindowClose = pyqtSignal()
    def __init__(self):
        super().__init__()  # 子窗口的实例化
        self.setupUi(self)

        self.pushButton_nextstep.clicked.connect(self.changePage)
        self.pushButton_2.clicked.connect(self.getPar)
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
        self.lineEdit_11.setValidator(doubleValidator)
        self.lineEdit_12.setValidator(doubleValidator)
        self.lineEdit_13.setValidator(doubleValidator)
        self.lineEdit_14.setValidator(doubleValidator)
        self.lineEdit_15.setValidator(doubleValidator)
        self.lineEdit_16.setValidator(doubleValidator)
        self.lineEdit_17.setValidator(doubleValidator)
        self.lineEdit_18.setValidator(doubleValidator)
        self.lineEdit_19.setValidator(doubleValidator)
        self.lineEdit_20.setValidator(doubleValidator)
        self.lineEdit_22.setValidator(doubleValidator)
        self.lineEdit_23.setValidator(doubleValidator)
        self.lineEdit_24.setValidator(doubleValidator)
        self.lineEdit_25.setValidator(doubleValidator)
        self.lineEdit_26.setValidator(doubleValidator)
        self.lineEdit_27.setValidator(doubleValidator)
        self.lineEdit_28.setValidator(doubleValidator)
        self.lineEdit_29.setValidator(doubleValidator)
        self.lineEdit_30.setValidator(doubleValidator)
        self.lineEdit_31.setValidator(doubleValidator)
        self.lineEdit_21.setValidator(doubleValidator)
        self.lineEdit_33.setValidator(doubleValidator)
        self.lineEdit_34.setValidator(doubleValidator)
        self.lineEdit_35.setValidator(doubleValidator)
        self.lineEdit_32.setValidator(doubleValidator)
        self.lineEdit_37.setValidator(doubleValidator)
        self.lineEdit_38.setValidator(doubleValidator)


    # 转到下一标签页
    def changePage(self):
        self.tabWidget.setCurrentIndex(1)

    # 获取参数
    def getPar(self):
        # 检查是否为空
        if(len(self.lineEdit.text()) == 0 or len(self.lineEdit_2.text()) == 0 or len(self.lineEdit_3.text()) == 0 or           
            len(self.lineEdit_4.text()) == 0 or len(self.lineEdit_5.text()) == 0 or len(self.lineEdit_6.text()) == 0 or           
            len(self.lineEdit_7.text()) == 0 or len(self.lineEdit_8.text()) == 0 or len(self.lineEdit_9.text()) == 0 or           
            len(self.lineEdit_10.text()) == 0 or len(self.lineEdit_11.text()) == 0 or len(self.lineEdit_12.text()) == 0 or           
            len(self.lineEdit_13.text()) == 0 or len(self.lineEdit_14.text()) == 0 or len(self.lineEdit_15.text()) == 0 or           
            len(self.lineEdit_16.text()) == 0 or len(self.lineEdit_17.text()) == 0 or len(self.lineEdit_18.text()) == 0 or 
            len(self.lineEdit_19.text()) == 0 or len(self.lineEdit_20.text()) == 0 or len(self.lineEdit_21.text()) == 0 or
            len(self.lineEdit_22.text()) == 0 or len(self.lineEdit_23.text()) == 0 or len(self.lineEdit_24.text()) == 0 or           
            len(self.lineEdit_25.text()) == 0 or len(self.lineEdit_26.text()) == 0 or len(self.lineEdit_27.text()) == 0 or           
            len(self.lineEdit_28.text()) == 0 or len(self.lineEdit_29.text()) == 0 or len(self.lineEdit_30.text()) == 0 or           
            len(self.lineEdit_31.text()) == 0 or len(self.lineEdit_33.text()) == 0 or len(self.lineEdit_37.text()) == 0 or     
            len(self.lineEdit_34.text()) == 0 or len(self.lineEdit_35.text()) == 0 or len(self.lineEdit_38.text()) == 0 or len(self.lineEdit_32.text()) == 0):
            # 显示提示框
            QMessageBox.warning(self, "警告", "有空值！")
        else:
            # 获取已知条件内容
            Ne = float(self.lineEdit.text())               # 核电厂输出电功率
            n_1 = float(self.lineEdit_2.text()) / 100      # 一回路能量利用系数
            x_fh = float(self.lineEdit_3.text()) / 100     # 蒸汽发生器出口蒸汽干度
            ks = float(self.lineEdit_4.text()) / 100       # 蒸汽发生器排污率
            n_hi = float(self.lineEdit_5.text()) / 100     # 高压缸内效率
            n_li = float(self.lineEdit_6.text()) / 100     # 低压缸内效率
            n_m = float(self.lineEdit_7.text()) / 100      # 汽轮机组机械效率
            n_ge = float(self.lineEdit_8.text()) / 100     # 发电机效率
            dP_fh = float(self.lineEdit_9.text()) / 100    # 新蒸汽压损，后期别忘了乘以SG的饱和压力
            dP_rh = float(self.lineEdit_10.text()) / 100   # 再热蒸汽压损，后期别忘了乘以p_hz
            dP_ej = float(self.lineEdit_11.text()) / 100   # 回热抽气压损，后期别忘了乘以p_ej
            dP_cd = float(self.lineEdit_12.text()) / 100   # 低压缸排气压损
            dP_f = float(self.lineEdit_13.text()) / 100    # 流动压损
            xt_hu = float(self.lineEdit_14.text())         # 高加出口端差
            xt_lu = float(self.lineEdit_15.text())         # 低加出口端差
            n_eNPP = float(self.lineEdit_21.text())        # 假定电厂效率
            G_cd = float(self.lineEdit_32.text())          # 假定冷凝器凝水量
            n_h = float(self.lineEdit_16.text()) / 100     # 加热器效率
            n_fwpp = float(self.lineEdit_17.text()) / 100  # 给水泵效率
            n_fwpti = float(self.lineEdit_18.text()) / 100 # 给水泵汽轮机内效率
            n_fwptm = float(self.lineEdit_19.text()) / 100 # 给水泵汽轮机机械效率
            n_fwptg = float(self.lineEdit_20.text()) / 100 # 给水泵汽轮机减速效率

            # 获取热力参数
            P_c = float(self.lineEdit_22.text())           # 反应堆冷却剂系统运行压力
            dT_sub = float(self.lineEdit_23.text())        # 反应堆出口冷却剂过冷度
            dT_c = float(self.lineEdit_24.text())          # 反应堆进出口冷却剂温升
            P_s = float(self.lineEdit_25.text())           # 蒸汽发生器饱和蒸汽压力
            dT_sw = float(self.lineEdit_26.text())         # 冷凝器中循环冷却水温升
            dT = float(self.lineEdit_27.text())            # 冷凝器传热端差
            T_sw1 = float(self.lineEdit_28.text())         # 循环冷却水温度
            dP_hz = float(self.lineEdit_29.text()) / 100   # 高压缸排气压力
            dT_rh = float(self.lineEdit_30.text())         # 第二级再热蒸汽出口温度
            Z = float(self.lineEdit_31.text())             # 回热级数
            Z_l = float(self.lineEdit_33.text())           # 低压给水加热器级数
            Z_h = float(self.lineEdit_34.text())           # 高压给水加热器级数
            dT_fw = float(self.lineEdit_35.text()) / 100   # 实际给水温度
            dP_fwpo = float(self.lineEdit_37.text())       # 给水泵出口压力(x倍GS二次侧蒸汽压力,MPa)
            dP_cwp = float(self.lineEdit_38.text())        # 凝水泵出口压力(x倍除氧器运行压力,MPa)

            # 生成一个json文件来展示储存输入参数
            jsontext_1 = {'已知条件':[], '给定热力参数':[]}
            jsontext_1['已知条件'].append({'name':'核电厂输出电功率(MW)', 'value':Ne})
            jsontext_1['已知条件'].append({'name':'一回路能量利用系数', 'value':n_1})
            jsontext_1['已知条件'].append({'name':'蒸汽发生器出口蒸汽干度', 'value':x_fh})
            jsontext_1['已知条件'].append({'name':'蒸汽发生器排污率', 'value':ks})
            jsontext_1['已知条件'].append({'name':'高压缸内效率', 'value':n_hi})
            jsontext_1['已知条件'].append({'name':'低压缸内效率', 'value':n_li})
            jsontext_1['已知条件'].append({'name':'汽轮机组机械效率', 'value':n_m})
            jsontext_1['已知条件'].append({'name':'发电机效率', 'value':n_ge})
            jsontext_1['已知条件'].append({'name':'新蒸汽压损(MPa)', 'value':dP_fh})
            jsontext_1['已知条件'].append({'name':'再热蒸汽压损(MPa)', 'value':dP_rh})
            jsontext_1['已知条件'].append({'name':'回热抽气压损(MPa)', 'value':dP_ej})
            jsontext_1['已知条件'].append({'name':'低压缸排气压损(kPa)', 'value':dP_cd})
            jsontext_1['已知条件'].append({'name':'流动压损(MPa)', 'value':dP_f})
            jsontext_1['已知条件'].append({'name':'高加出口端差(℃)', 'value':xt_hu})
            jsontext_1['已知条件'].append({'name':'低加出口端差(℃)', 'value':xt_lu})
            jsontext_1['已知条件'].append({'name':'假定电厂效率', 'value':n_eNPP})
            jsontext_1['已知条件'].append({'name':'假定冷凝器凝水量(kg/s)', 'value':G_cd})
            jsontext_1['已知条件'].append({'name':'加热器效率', 'value':n_h})
            jsontext_1['已知条件'].append({'name':'给水泵效率', 'value':n_fwpp})
            jsontext_1['已知条件'].append({'name':'给水泵汽轮机内效率', 'value':n_fwpti})
            jsontext_1['已知条件'].append({'name':'给水泵汽轮机机械效率', 'value':n_fwptm})
            jsontext_1['已知条件'].append({'name':'给水泵汽轮机减速效率', 'value':n_fwptg})
            #jsontext_2 = {'给定热力参数':[]}
            jsontext_1['给定热力参数'].append({'name':'反应堆冷却剂系统运行压力(MPa)', 'value':P_c})
            jsontext_1['给定热力参数'].append({'name':'反应堆出口冷却剂过冷度(℃)', 'value':dT_sub})
            jsontext_1['给定热力参数'].append({'name':'反应堆进出口冷却剂温升(℃)', 'value':dT_c})
            jsontext_1['给定热力参数'].append({'name':'蒸汽发生器饱和蒸汽压力(MPa)', 'value':P_s})
            jsontext_1['给定热力参数'].append({'name':'冷凝器中循环冷却水温升(℃)', 'value':dT_sw})
            jsontext_1['给定热力参数'].append({'name':'冷凝器传热端差(℃)', 'value':dT})
            jsontext_1['给定热力参数'].append({'name':'循环冷却水温度(℃)', 'value':T_sw1})
            jsontext_1['给定热力参数'].append({'name':'高压缸排气压力(MPa)', 'value':dP_hz})
            jsontext_1['给定热力参数'].append({'name':'第二级再热蒸汽出口温度(℃)', 'value':dT_rh})
            jsontext_1['给定热力参数'].append({'name':'回热级数', 'value':Z})
            jsontext_1['给定热力参数'].append({'name':'低压给水加热器级数', 'value':Z_l})
            jsontext_1['给定热力参数'].append({'name':'高压给水加热器级数', 'value':Z_h})
            jsontext_1['给定热力参数'].append({'name':'实际给水温度(℃)', 'value':dT_fw})
            jsontext_1['给定热力参数'].append({'name':'给水泵出口压力(x倍GS二次侧蒸汽压力,MPa)', 'value':dP_fwpo})
            jsontext_1['给定热力参数'].append({'name':'凝水泵出口压力(x倍除氧器运行压力,MPa)', 'value':dP_cwp})

            jsondata = json.dumps(jsontext_1, indent=4, separators=(',',':'), ensure_ascii=False)
            mkpath = global_var.get_value('mkpath')
            fileJson = open(mkpath + '/' + 'parameters.json', 'w', encoding='utf-8')
            fileJson.write(jsondata)
            fileJson.close()
            global_var.set_value('fileJson', fileJson.name)

            self.sendSignal()

            self.close()
    def sendSignal(self):
        self.parWindowClose.emit()
