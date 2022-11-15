import os 
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QObject, pyqtSignal
from Ui_parTab import Ui_parTab
from Ui_exist import Ui_exist
import json
import global_var

class parTabWindow(QWidget, Ui_parTab):
   outputSignal = pyqtSignal()
   def __init__(self):
      super().__init__()  # 子窗口的实例化
      self.setupUi(self)
      self.loadJson()
      self.pushButton_save.clicked.connect(self.savePar)
      self.pushButton_run.clicked.connect(self.run)

   # 解析json文件，并将内容显示在tableWidget中
   def loadJson(self):
      fileJson = global_var.get_value('fileJson')
      with open(fileJson, 'r', encoding='utf-8') as fp:
         data = json.load(fp)
      list1 = data['已知条件']
      list2 = data['给定热力参数']
      # 获取已知条件
      for i, item in enumerate(list1):
         self.parTable_left.setRowCount(self.parTable_left.rowCount() + 1)
         fname = QTableWidgetItem(item['name'])
         fname.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable) # 可选择但不可编辑
         self.parTable_left.setItem(self.parTable_left.rowCount() - 1, 0, fname)

         fvalue = QTableWidgetItem(str(item['value']))
         self.parTable_left.setItem(self.parTable_left.rowCount() - 1, 1, fvalue)

      # 获取给定热力参数
      for i, item in enumerate(list2):
         self.parTable_right.setRowCount(self.parTable_right.rowCount() + 1)
         fname = QTableWidgetItem(item['name'])
         fname.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable) # 可选择但不可编辑
         self.parTable_right.setItem(self.parTable_right.rowCount() - 1, 0, fname)

         fvalue = QTableWidgetItem(str(item['value']))
         self.parTable_right.setItem(self.parTable_right.rowCount() - 1, 1, fvalue)
      
   # 对于输入参数，显示可修改的表格，并支持保存（覆盖或另存）
   def savePar(self):
      jsontext_1 = {'已知条件':[], '给定热力参数':[]}
      for i in range(self.parTable_left.rowCount()):
         jsontext_1['已知条件'].append({'name':self.parTable_left.item(i, 0).text(), 'value':self.parTable_left.item(i, 1).text()})
      
      for i in range(self.parTable_right.rowCount()):
         jsontext_1['给定热力参数'].append({'name':self.parTable_right.item(i, 0).text(), 'value':self.parTable_right.item(i, 1).text()})
      
      jsondata = json.dumps(jsontext_1, indent=4, separators=(',',':'), ensure_ascii=False)
      fileJson = global_var.get_value('fileJson')
      exist = os.path.exists(fileJson)
      if(exist):
         existWindow = Exist()
         existWindow.exec()   # 为啥用exec()参考: https://wanzhou.blog.csdn.net/article/details/115027549?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-1.essearch_pc_relevant&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-1.essearch_pc_relevant
         flag = global_var.get_value('overlay')
         if(flag == 1):
            fileJson = open(fileJson, 'w', encoding='utf-8')
            fileJson.write(jsondata)
            fileJson.close()
         elif(flag == 2):
            fileJson, ok = QFileDialog.getSaveFileName(self, "保存输入参数文件", "C:/", "json文件 (*.json)")
            global_var.set_value('fileJson', fileJson)   # 确保全局变量fileJson永远都是最新的（也就是当前表格中的）
            fileJson = open(fileJson, 'w', encoding='utf-8')
            fileJson.write(jsondata)
            fileJson.close()
      else:
         fileJson = open(fileJson, 'w', encoding='utf-8')
         fileJson.write(jsondata)
         fileJson.close()
   # 对于输入参数有“运行”按钮，用于生成输出数据，以表格形式展示主窗口在一个新的tab上
   def run(self):
      fileJson = global_var.get_value('fileJson')
      # dicPath = os.path.split(fileJson)[0]
      dicPath = fileJson.split(".json")[0]
      # 解析Json文件
      with open(fileJson, 'r', encoding='utf-8') as fp:
         data = json.load(fp)
      list1 = data['已知条件']
      list2 = data['给定热力参数']
      # 获取已知条件
      for i, item in enumerate(list1):
         fname = item['name']
         if fname == "核电厂输出电功率(MW)": Ne = float(item['value'])
         elif fname == "一回路能量利用系数":   n_1 = float(item['value'])
         elif fname == "蒸汽发生器出口蒸汽干度":  x_fh = float(item['value'])
         elif fname == "蒸汽发生器排污率":  ks = float(item['value'])
         elif fname == "高压缸内效率":   n_hi = float(item['value'])
         elif fname == "低压缸内效率":   n_li = float(item['value'])
         elif fname == "汽轮机组机械效率":  n_m = float(item['value'])
         elif fname == "发电机效率":  n_ge = float(item['value'])
         elif fname == "新蒸汽压损(MPa)":   dP_fh = float(item['value'])
         elif fname == "再热蒸汽压损(MPa)": dP_rh = float(item['value'])
         elif fname == "回热抽气压损(MPa)": dP_ej = float(item['value'])
         elif fname == "低压缸排气压损(kPa)":  dp_cd = float(item['value'])
         elif fname == "流动压损(MPa)":  dP_f = float(item['value'])
         elif fname == "高加出口端差(℃)":   xt_hu = float(item['value'])
         elif fname == "低加出口端差(℃)":   xt_lu = float(item['value'])
         elif fname == "假定电厂效率":  n_eNPP = float(item['value'])
         elif fname == "假定冷凝器凝水量(kg/s)":  G_cd = float(item['value'])
         elif fname == "加热器效率":  n_h = float(item['value'])
         elif fname == "给水泵效率":  n_fwpp = float(item['value'])
         elif fname == "给水泵汽轮机内效率":   n_fwpti = float(item['value'])
         elif fname == "给水泵汽轮机机械效率": n_fwptm = float(item['value'])
         elif fname == "给水泵汽轮机减速效率": n_fwptg = float(item['value'])
      # 获取给定热力参数
      for i, item in enumerate(list2):
         fname = item['name']
         if fname == "反应堆冷却剂系统运行压力(MPa)": P_c = float(item['value'])
         elif fname == "反应堆出口冷却剂过冷度(℃)": dT_sub = float(item['value'])
         elif fname == "反应堆进出口冷却剂温升(℃)": dT_c = float(item['value'])
         elif fname == "蒸汽发生器饱和蒸汽压力(MPa)": P_s = float(item['value'])
         elif fname == "冷凝器中循环冷却水温升(℃)": dT_sw = float(item['value'])
         elif fname == "冷凝器传热端差(℃)": dT = float(item['value'])
         elif fname == "循环冷却水温度(℃)": T_sw1 = float(item['value'])
         elif fname == "高压缸排气压力(MPa)": dP_hz = float(item['value'])
         elif fname == "第二级再热蒸汽出口温度(℃)": dT_rh = float(item['value'])
         elif fname == "回热级数": Z = float(item['value'])
         elif fname == "低压给水加热器级数": Z_l = float(item['value'])
         elif fname == "高压给水加热器级数": Z_h = float(item['value'])
         elif fname == "实际给水温度(℃)": dT_fw = float(item['value'])
         elif fname == "给水泵出口压力(x倍GS二次侧蒸汽压力,MPa)": dP_fwpo = float(item['value'])
         elif fname == "凝水泵出口压力(x倍除氧器运行压力,MPa)": dP_cwp = float(item['value'])
         
      # 将数据都存为全局变量，用于后续生成.py文件
      global_var.set_value('Ne', Ne)
      global_var.set_value('n_1', n_1)
      global_var.set_value('x_fh', x_fh)
      global_var.set_value('ks', ks)
      global_var.set_value('n_hi', n_hi)
      global_var.set_value('n_li', n_li)
      global_var.set_value('n_m', n_m)
      global_var.set_value('n_ge', n_ge)
      global_var.set_value('dP_fh', dP_fh)
      global_var.set_value('dP_rh', dP_rh)
      global_var.set_value('dP_ej', dP_ej)
      global_var.set_value('dp_cd', dp_cd)
      global_var.set_value('dP_f', dP_f)
      global_var.set_value('xt_hu', xt_hu)
      global_var.set_value('xt_lu', xt_lu)
      global_var.set_value('n_eNPP', n_eNPP)
      global_var.set_value('G_cd', G_cd)
      global_var.set_value('n_h', n_h)
      global_var.set_value('n_fwpp', n_fwpp)
      global_var.set_value('n_fwpti', n_fwpti)
      global_var.set_value('n_fwptm', n_fwptm)
      global_var.set_value('n_fwptg', n_fwptg)

      global_var.set_value('P_c', P_c)
      global_var.set_value('dT_sub', dT_sub)
      global_var.set_value('dT_c', dT_c)
      global_var.set_value('P_s', P_s)
      global_var.set_value('dT_sw', dT_sw)
      global_var.set_value('dT', dT)
      global_var.set_value('T_sw1', T_sw1)
      global_var.set_value('dP_hz', dP_hz)
      global_var.set_value('dT_rh', dT_rh)
      global_var.set_value('Z', Z)
      global_var.set_value('Z_l', Z_l)
      global_var.set_value('Z_h', Z_h)
      global_var.set_value('dT_fw', dT_fw)
      global_var.set_value('dP_fwpo', dP_fwpo)
      global_var.set_value('dP_cwp', dP_cwp)



      # 核心程序
      from iapws import iapws97 as ip
      import math as m
      # 一回路冷却剂参数
      T_cs = ip.IAPWS97(P=P_c, x=0).T - 273.15  # 工作压力对应饱和温度
      T_co = T_cs - dT_sub  # 反应堆出口冷却剂温度
      T_ci = T_co - dT_c  # 反应堆进口冷却剂温度

      # 蒸汽初参数
      T_s = ip.IAPWS97(P=P_s, x=0).T - 273.15  # 对应的饱和温度
      T_fh = ip.IAPWS97(P=P_s, x=x_fh).T - 273.15  # 新蒸汽温度
      h_fh = ip.IAPWS97(P=P_s, x=x_fh).h  # 新蒸汽比焓
      s_fh = ip.IAPWS97(P=P_s, x=x_fh).s  # 新蒸汽比熵
      dT_m = (T_co-T_ci)/m.log((T_co-T_s)/(T_ci-T_s))  # 对数平均传热温差

      # 蒸汽终参数
      T_cd = T_sw1+dT_sw+dT  # 凝结水温度
      P_cd = ip.IAPWS97(T=T_cd+273.15, x=0).P  # 凝结水压力
      h_cd = ip.IAPWS97(T=T_cd+273.15, x=0).h  # 凝结水比焓

      # 高压缸参数
      dP_fh = dP_fh * P_s  # 新蒸汽压损
      P_hi = P_s - dP_fh  # 高压缸进口蒸汽压力
      h_hi = h_fh  # 高压缸进口蒸汽比焓，定焓过程
      x_hi = ip.IAPWS97(P=P_hi, h=h_hi).x  # 进口蒸汽干度
      s_hi = ip.IAPWS97(P=P_hi, h=h_hi).s  # 进口蒸汽比熵
      P_hz = dP_hz * P_hi  # 排气压力，最佳分压12~14%
      h_hzs = ip.IAPWS97(P=P_hz, s=s_hi).h  # 高压缸排气理想比焓
      h_hz = h_hi - n_m*n_hi*(h_hi - h_hzs)  # 高压缸排气实际比焓
      x_hz = ip.IAPWS97(P=P_hz, h=h_hz).x  # 排气干度

      # 蒸汽中间再热参数
      # 在汽水分离器再热器中的总压降为高压缸排汽压力的3%左右。
      # 为计算方便，假设高压缸排汽经过汽水分离再热系统时各设备的压降相同，
      # 均为总压降的1/3。参照大亚湾的蒸汽参数，汽水分离器除去蒸汽中98%的水
      dP_rh = dP_rh * P_hz  # 再热蒸汽压损
      P_spi = P_hz  # 汽水分离器进口蒸汽压力
      x_spi = x_hz  # 汽水分离器进口蒸汽干度
      P_uw = 0.99*P_hz  # 汽水分离器出口疏水压力
      h_uw = ip.IAPWS97(P=P_uw, x=0).h  # 汽水分离器出口疏水比焓

      # 一级再热器
      P_rh1i = 0.99*P_hz  # 一级再热器进口蒸汽压力
      x_rh1i = x_spi/(1-0.98*(1-x_spi))  # 一级再热器进口蒸汽干度
      h_rh1i = ip.IAPWS97(P=P_rh1i, x=x_rh1i).h  # 一级再热器进口蒸汽比焓

      # 二级再热器
      P_rh2i = 0.98*P_hz  # 再热蒸汽进口压力
      P_rh2z = 0.97*P_hz  # 二级再热器出口压力
      T_rh2z = T_fh - dT_rh  # 二级再热器出口温度
      h_rh2z = ip.IAPWS97(P=P_rh2z, T=T_rh2z+273.15).h  # 二级再热器出口蒸汽比焓
      dh_rh = (h_rh2z - h_rh1i)/2  # 每级再热器平均焓升
      h_rh1z = h_rh1i+dh_rh  # 一级再热器出口蒸汽比焓
      h_rh2i = h_rh1z  # 二级再热器进口蒸汽比焓
      T_rh2i = ip.IAPWS97(P=P_rh2i, h=h_rh2i).T-273.15  # 二级再热器进口蒸汽温度
      P_rh2hs = P_hi  # 加热蒸汽进口压力
      x_rh2hs = x_hi  # 加热蒸汽进口干度

      # 低压缸参数
      # 考虑低压缸的进汽损失占再热器出口压力的2%
      P_li = dP_f*P_rh2z  # 低压缸进气压力，考虑1%的压损
      h_li = h_rh2z  # 低压缸进口进气比焓，考虑为定焓过程
      T_li = ip.IAPWS97(P=P_li, h=h_li).T - 273.15  # 进口蒸汽温度
      dp_cd = (1/(1-dp_cd)-1)*P_cd  # 低压缸排气压损
      P_lz = P_cd + dp_cd  # 低压缸排气压力
      s_li = ip.IAPWS97(P=P_li, h=h_li).s  # 进口蒸汽比熵
      h_lzs = ip.IAPWS97(P=P_lz, s=s_li).h  # 低压缸排气理想比焓，考虑为定熵过程
      h_lz = h_rh2z - n_li*(h_rh2z - h_lzs)  # 排气实际比焓
      x_lz = ip.IAPWS97(P=P_lz, h=h_lz).x  # 排气干度

      # 给水的焓升分配
      h_s = ip.IAPWS97(P=P_s, x=0).h  # GS工作压力下的饱和水焓
      h_cd = ip.IAPWS97(T=T_cd+273.15, x=0).h  # 冷凝器出口凝结水比焓
      dh_fwop = (h_s - h_cd)/(Z+1)  # 理论给水焓升
      h_fwop = h_cd+Z*dh_fwop  # GS最佳给水比焓
      T_fwop = ip.IAPWS97(P=P_s, h=h_fwop).T-273.15  # 最佳给水温度
      T_fw = dT_fw * T_fwop  # 实际给水温度
      h_fw = ip.IAPWS97(P=P_s, T=T_fw+273.15).h  # 实际给水比焓
      dh_fw = (h_fw - h_cd)/Z  # 第一次给水回热分配
      # 除氧器
      P_dea = P_uw  # 除氧器运行压力，等于汽水分离器的疏水压力
      h_deao = ip.IAPWS97(P=P_dea, x=0).h  # 除氧器对应饱和水比焓
      dh_fwh = (h_fw - h_deao)/Z_h  # 高压给水加热器给水焓升
      dh_fwl = (h_deao - h_cd)/(Z_l+1)  # 除氧器及低压给水加热器给水焓升

      # 给水回热系统中的压力选择
      P_cwp = dP_cwp * P_dea  # 取凝水泵出口压力为除氧器运行压力的3倍
      h_cwp = h_cd  # 凝水泵出口给水比焓
      dP_cws = P_cwp - P_dea  # 凝水泵出口至除氧器的阻力压降
      dP_fi = dP_cws/(Z_l+1)  # 每级低压加热器及除氧器的平均压降

      # 一级低压给水加热器
      P_fw1i = P_cwp  # 进口给水压力
      h_fw1i = h_cwp  # 进口给水比焓
      T_fw1i = ip.IAPWS97(P=P_fw1i, h=h_fw1i).T-273.15  # 进口给水温度
      P_fw1o = P_fw1i - dP_fi  # 出口给水压力
      h_fw1o = h_fw1i+dh_fwl  # 出口给水比焓
      T_fw1o = ip.IAPWS97(P=P_fw1o, h=h_fw1o).T-273.15  # 出口给水温度
      T_ro1k = T_fw1o+xt_lu  # 汽侧疏水温度
      h_ro1k = ip.IAPWS97(T=T_ro1k+273.15, x=0).h  # 汽侧疏水比焓
      P_ro1k = ip.IAPWS97(T=T_ro1k+273.15, x=0).P  # 汽侧压力
      # 二级低压给水加热器
      P_fw2i = P_fw1o
      h_fw2i = h_fw1o
      T_fw2i = T_fw1o
      P_fw2o = P_fw2i - dP_fi
      h_fw2o = h_fw2i+dh_fwl
      T_fw2o = ip.IAPWS97(P=P_fw2o, h=h_fw2o).T - 273.15
      T_ro2k = T_fw2o+xt_lu
      h_ro2k = ip.IAPWS97(T=T_ro2k+273.15, x=0).h
      P_ro2k = ip.IAPWS97(T=T_ro2k+273.15, x=0).P
      # 三级低压给水加热器
      P_fw3i = P_fw2o
      h_fw3i = h_fw2o
      T_fw3i = T_fw2o
      P_fw3o = P_fw3i - dP_fi
      h_fw3o = h_fw3i+dh_fwl
      T_fw3o = ip.IAPWS97(P=P_fw3o, h=h_fw3o).T - 273.15
      T_ro3k = T_fw3o+xt_lu
      h_ro3k = ip.IAPWS97(T=T_ro3k+273.15, x=0).h
      P_ro3k = ip.IAPWS97(T=T_ro3k+273.15, x=0).P
      # 四级低压给水加热器
      P_fw4i = P_fw3o
      h_fw4i = h_fw3o
      T_fw4i = T_fw3o
      P_fw4o = P_fw4i - dP_fi
      h_fw4o = h_fw4i+dh_fwl
      T_fw4o = ip.IAPWS97(P=P_fw4o, h=h_fw4o).T - 273.15
      T_ro4k = T_fw4o+xt_lu
      h_ro4k = ip.IAPWS97(T=T_ro4k+273.15, x=0).h
      P_ro4k = ip.IAPWS97(T=T_ro4k+273.15, x=0).P

      # 除氧器
      P_dea = P_uw  # 运行压力
      h_deai = h_fw4o  # 进口给水比焓
      h_deao = h_deai+dh_fw  # 出口给水比焓
      T_dea = ip.IAPWS97(P=P_dea, x=0).T - 273.15  # 出口给水温度
      P_fwpo = dP_fwpo*P_s  # 给水泵出口压力
      h_fwpo = h_deao  # 给水泵出口流体比焓
      P_fwi = P_s + 0.1  # GS二次侧进口给水压力

      # 六级高压给水加热器
      P_fw6i = P_fwpo
      h_fw6i = h_fwpo
      T_fw6i = ip.IAPWS97(P=P_fw6i, h=h_fw6i).T - 273.15
      P_fw6o = P_fw6i - (P_fw6i - P_fwi)/2
      h_fw6o = h_fw6i + dh_fwh
      T_fw6o = ip.IAPWS97(P=P_fw6o, h=h_fw6o).T - 273.15
      T_ro6k = T_fw6o + xt_hu  # 汽侧疏水温度
      h_ro6k = ip.IAPWS97(T=T_ro6k+273.15, x=0).h
      P_ro6k = ip.IAPWS97(T=T_ro6k+273.15, x=0).P
      # 七级高压给水加热器
      P_fw7i = P_fw6o
      h_fw7i = h_fw6o
      T_fw7i = T_fw6o
      P_fw7o = P_fwi
      h_fw7o = h_fw7i + dh_fwh
      T_fw7o = ip.IAPWS97(P=P_fw7o, h=h_fw7o).T - 273.15
      T_ro7k = T_fw7o + xt_hu
      h_ro7k = ip.IAPWS97(T=T_ro7k+273.15, x=0).h
      P_ro7k = ip.IAPWS97(T=T_ro7k+273.15, x=0).P

      # 高压缸抽汽
      # 六级给水加热器抽汽参数
      dP_e6 = (1/(1-dP_ej)-1)*P_ro6k  # 压损
      P_hes6 = P_ro6k+dP_e6  # 抽汽压力
      h_hes6s = ip.IAPWS97(P=P_hes6, s=s_hi).h  # 抽汽理想比焓
      h_hes6 = h_hi - n_m*n_hi*(h_hi - h_hes6s)  # 抽汽比焓
      x_hes6 = ip.IAPWS97(P=P_hes6, h=h_hes6).x  # 抽汽干度
      # 七级给水加热器抽汽参数
      dP_e7 = (1/(1-dP_ej)-1)*P_ro7k
      P_hes7 = P_ro7k + dP_e7
      h_hes7s = ip.IAPWS97(P=P_hes7, s=s_hi).h
      h_hes7 = h_hi - n_m*n_hi*(h_hi - h_hes7s)
      x_hes7 = ip.IAPWS97(P=P_hes7, h=h_hes7).x

      # 低压缸抽汽
      # 一级给水加热器抽汽参数
      dP_e1 = (1/(1-dP_ej)-1)*P_ro1k
      P_les1 = P_ro1k+dP_e1
      h_les1s = ip.IAPWS97(P=P_les1, s=s_li).h
      h_les1 = h_li - n_m*n_li*(h_li - h_les1s)
      x_les1 = ip.IAPWS97(P=P_les1, h=h_les1).x
      # 二级给水加热器抽汽参数
      dP_e2 = (1/(1-dP_ej)-1)*P_ro2k
      P_les2 = P_ro2k+dP_e2
      h_les2s = ip.IAPWS97(P=P_les2, s=s_li).h
      h_les2 = h_li - n_m*n_li*(h_li - h_les2s)
      x_les2 = ip.IAPWS97(P=P_les2, h=h_les2).x
      # 三级给水加热器抽汽参数
      dP_e3 = (1/(1-dP_ej)-1)*P_ro3k
      P_les3 = P_ro3k+dP_e3
      h_les3s = ip.IAPWS97(P=P_les3, s=s_li).h
      h_les3 = h_li - n_m*n_li*(h_li - h_les3s)
      x_les3 = ip.IAPWS97(P=P_les3, h=h_les3).x
      # 四级给水加热器抽汽参数
      dP_e4 = (1/(1-dP_ej)-1)*P_ro4k
      P_les4 = P_ro4k+dP_e4
      h_les4s = ip.IAPWS97(P=P_les4, s=s_li).h
      h_les4 = h_li - n_m*n_li*(h_li - h_les4s)
      x_les4 = ip.IAPWS97(P=P_les4, h=h_les4).x

      # 再热器抽汽
      # 一级再热器抽汽参数
      P_rh1 = P_hes7  # 加热蒸汽进口压力
      x_rh1 = x_hes7  # 加热蒸汽进口干度
      T_rh1 = ip.IAPWS97(P=P_rh1, x=x_rh1).T - 273.15  # 加热蒸汽进口温度
      h_rh1 = ip.IAPWS97(P=P_rh1, x=x_rh1).h  # 加热蒸汽进口比焓
      h_zs1 = ip.IAPWS97(P=P_rh1, x=0).h  # 再热器疏水比焓
      # 二级再热器抽汽参数
      P_rh2 = P_hi
      x_rh2 = x_hi
      T_rh2 = ip.IAPWS97(P=P_rh2, x=x_rh2).T - 273.15  # 加热蒸汽进口温度
      h_rh2 = ip.IAPWS97(P=P_rh2, x=x_rh2).h  # 加热蒸汽进口比焓
      h_zs2 = ip.IAPWS97(P=P_rh2, x=0).h  # 再热器疏水比焓

      # 蒸汽发生器总蒸汽产量的计算
      h_a = h_hi-ip.IAPWS97(P=1.05*P_cd, x=0).h  # 给水泵汽轮机中蒸汽的绝热焓降
      for i in range(0, 10**99):
         QR = Ne / n_eNPP  # 反应堆功率
         Ds = 1000*QR*n_1/((h_fh - h_s)+(1+ks)*(h_s - h_fw))  # GS的蒸汽产量
         G_fw = (1+ks)*Ds  # GS给水流量
         for j in range(0, 10**99):
            H_fwp = P_fwpo - P_dea  # 给水泵的扬程
            # 给水泵中给水的密度,定为给水泵密度进出口平均值
            rho_fwp = 0.5*(ip.IAPWS97(P=P_dea, x=0).rho +
                           ip.IAPWS97(P=P_fwpo, x=0).rho)
            N_fwpp = G_fw*H_fwp/rho_fwp  # 给水泵有效输出功率
            N_fwpt = N_fwpp/(n_fwpp*n_fwpti*n_fwptm*n_fwptg)  # 给水泵汽轮机理论功率
            G_fwps = 1000*N_fwpt/h_a  # 给水泵汽轮机耗气量
            # 低压给水加热器抽汽量
            G_les4 = G_cd*(h_fw4o - h_fw4i)/(n_h*(h_les4 - h_ro4k))  # 第四级抽汽量
            G_les3 = (G_cd*(h_fw3o - h_fw3i)-n_h*G_les4 *
                        (h_ro4k-h_ro3k))/(n_h*(h_les3 - h_ro3k))
            G_les2 = (G_cd*(h_fw2o - h_fw2i)-n_h*(G_les3+G_les4)
                        * (h_ro3k - h_ro2k))/(n_h*(h_les2 - h_ro2k))
            G_les1 = (G_cd*(h_fw1o - h_fw1i)-n_h*(G_les2+G_les3+G_les4)
                        * (h_ro2k - h_ro1k))/(n_h*(h_les1 - h_ro1k))
            # 低压缸耗气量
            G_sl = (0.6*1000*Ne/(n_m*n_ge)+G_les4*(h_les4-h_lz)+G_les3 *
                     (h_les3-h_lz)+G_les2*(h_les2-h_lz)+G_les1*(h_les1-h_lz))/(h_li-h_lz)
            # 再热器加热蒸汽量
            G_zc1 = G_sl*dh_rh/(n_h*(h_rh1 - h_zs1))
            G_zc2 = G_sl*dh_rh/(n_h*(h_rh2 - h_zs2))
            # 高压给水加热器抽汽量
            G_hes7 = (G_fw*(h_fw7o - h_fw7i) - n_h*G_zc2 *
                        (h_zs2 - h_ro7k))/(n_h*(h_hes7 - h_ro7k))
            G_hes6 = (G_fw*(h_fw6o - h_fw6i) - n_h*G_zc1*(h_zs1 - h_ro6k) -
                        n_h*(G_zc2+G_hes7)*(h_ro7k - h_ro6k))/(n_h*(h_hes6 - h_ro6k))
            # 汽水分离器疏水流量
            G_uw = G_sl*(x_rh1i - x_spi)/x_spi
            G_h1 = G_sl+G_uw
            # 除氧器耗气量
            G_sdea = (G_fw*h_deao - G_uw*h_uw - G_cd*h_fw4o -
                        (G_zc1+G_zc2+G_hes6+G_hes7)*h_ro6k)/h_hz
            # 高压缸出口排气总流量
            G_h = G_h1+G_sl+G_uw
            # 高压缸耗气量
            G_sh = (0.4*1000*Ne/(n_m*n_ge)+G_hes7*(h_hes7 - h_hz)+G_hes6 *
                     (h_hes6 - h_hz)+G_zc1*(h_rh1 - h_hz))/(h_hi - h_hz)
            # 对假设冷凝水流量验证
            Ds = G_fwps+G_zc2+G_sh  # 对应的新蒸汽耗量
            G_fw1 = (1+ks)*Ds  # 给水流量
            G_cd1 = G_fw1 - G_sdea - G_uw - (G_hes6+G_hes7+G_zc1+G_zc2)
            if abs(G_cd1 - G_cd)/G_cd < 1e-2:
                  break
            else:
                  G_cd = G_cd1
                  G_fw = G_fw1
         QR = (Ds*(h_fh - h_fw)+ks*Ds*(h_s - h_fw))/(1000*n_1)
         n_eNPP1 = Ne/QR
         # 输出结果
         data1 = open(dicPath + "_" + "interation.txt", 'a', encoding='utf-8')
         print('-----', i, '------', file=data1)
         print('1.核电厂效率ηnNPP = ', n_eNPP, file=data1)
         print('2.反应堆热功率QR = ', QR, file=data1)
         print('3.蒸汽发生器总蒸汽产量Ds = ', Ds, file=data1)
         print('4.汽轮机高压缸耗气量Gshp = ', G_sh, file=data1)
         print('5.汽轮机低压缸耗气量Gslp = ', G_sl, file=data1)
         print('6.第一级再热器耗气量Gsrh1 = ', G_zc1, file=data1)
         print('7.第二级再热器耗气量Gsrh2 = ', G_zc2, file=data1)
         print('8.除氧器耗气量Gsdea = ', G_sdea, file=data1)
         print('9.给水泵汽轮机耗气量Gsfwp = ', G_fwps, file=data1)
         print('10.给水泵给水量Gfw = ', G_fw, file=data1)
         print('11.给水泵扬程Hfwp = ', H_fwp, file=data1)
         print('12.高压缸抽汽量', file=data1)
         print(' 12.1.第七级抽汽量Ghes7 = ', G_hes7, file=data1)
         print(' 12.2.第六级抽汽量Ghes6 = ', G_hes6, file=data1)
         print('13.低压缸抽汽量', file=data1)
         print(' 13.1.第四级抽汽量Gles4 = ', G_les4, file=data1)
         print(' 13.2.第三级抽汽量Gles3 = ', G_les3, file=data1)
         print(' 13.3.第二级抽汽量Gles2 = ', G_les2, file=data1)
         print(' 13.4.第一级抽汽量Gles1 = ', G_les1, file=data1)
         print('14.凝结水量Gcd = ', G_cd, file=data1)
         print('15.汽水分离器疏水量Guw = ', G_uw, file=data1)
         print('16.一级再热器加热蒸汽量Gzc1 = ', G_zc1, file=data1)
         print('17.二级再热器加热蒸汽量Gzc2 = ', G_zc2, file=data1)
         data1.close()
         print("done")
         if abs(n_eNPP - n_eNPP1)/n_eNPP < 1e-3:
            break
         else:
            n_eNPP = n_eNPP1

      data = open(dicPath + "_" + "data.txt", 'w', encoding='utf-8')
      print('----------附表1---------', file=data)
      print('1.核电厂输出功率Ne = ', Ne, file=data)
      print('2.一回路能量利用系数η1 = ', n_1, file=data)
      print('3.蒸汽发生器出口蒸汽干度Xfh = ', x_fh, file=data)
      print('4.蒸汽发生器排污率ξd = ', ks, file=data)
      print('5.高压缸内效率ηhi = ', n_hi, file=data)
      print('6.低压缸内效率ηli = ', n_li, file=data)
      print('7.汽轮机组机械效率ηm = ', n_m, file=data)
      print('8.发电机效率ηge = ', n_ge, file=data)
      print('9.新蒸汽压损Δpfh = ', dP_fh, file=data)
      print('10.再热蒸汽压损Δprh = ', dP_rh, file=data)
      print('11.回热蒸汽压损Δpej = 4%pej', file=data)
      print('12.低压缸排气压损Δpcd = ', dp_cd, file=data)
      print('13.高压给水加热器出口端差θhu = ', xt_hu, file=data)
      print('14.低压给水加热器出口端差θlu = ', xt_lu, file=data)
      print('15.加热器效率ηh = ', n_h, file=data)
      print('16.给水泵效率ηfwpp = ', n_fwpp, file=data)
      print('17.给水泵汽轮机内效率ηfwpti = ', n_fwpti, file=data)
      print('18.给水泵汽轮机机械效率ηfwptm = ', n_fwptm, file=data)
      print('19.给水泵汽轮机减速器效率ηfwptg = ', n_fwptg, file=data)
      print('20.循环冷却水进口温度Tsw1 = ', T_sw1, file=data)
      print('----------附表2---------', file=data)
      print('1.反应堆冷却剂系统运行压力pc = ', P_c, file=data)
      print('2.冷却剂压力对应的饱和温度Tcs = ', T_cs, file=data)
      print('3.反应堆出口冷却剂过冷度ΔTsub = ', dT_sub, file=data)
      print('4.反应堆出口冷却剂温度Tco = ', T_co, file=data)
      print('5.反应堆进出口冷却剂温升ΔTc = ', dT_c, file=data)
      print('6.反应堆进口冷却剂温度Tci = ', T_ci, file=data)
      print('7.蒸汽发生器饱和蒸汽压力ps = ', P_s, file=data)
      print('8.蒸汽发生器饱和蒸汽温度Tfh = ', T_fh, file=data)
      print('9.一、二次侧对数平均温差ΔTm = ', dT_m, file=data)
      print('10.冷凝器中循环冷却水温升ΔTsw = ', dT_sw, file=data)
      print('11.冷凝器传热端差δT = ', dT, file=data)
      print('12.冷凝器凝结水饱和温度Tcd = ', T_cd, file=data)
      print('13.冷凝器的运行压力pcd = ', P_cd, file=data)
      print('14.高压缸进口的蒸汽压力phi = ', P_hi, file=data)
      print('15.高压缸进口蒸汽干度Xhi = ', x_hi, file=data)
      print(' 15.1.蒸汽发生器出口蒸汽比焓hfh = ', h_fh, file=data)
      print(' 15.2.蒸汽发生器出口蒸汽比熵sfh = ', s_fh, file=data)
      print(' 15.3.高压缸进口蒸汽比熵shi = ', s_hi, file=data)
      print('16.高压缸排气压力phz = ', P_hz, file=data)
      print('17.高压缸排气干度Xhz = ', x_hz, file=data)
      print(' 17.1.高压缸进口蒸汽比焓hhi = ', h_hi, file=data)
      print(' 17.2.高压缸出口理想比焓hhzs = ', h_hzs, file=data)
      print(' 17.3.高压缸出口蒸汽比焓hhz = ', h_hz, file=data)
      print('18.汽水分离器进口蒸汽压力pspi = ', P_spi, file=data)
      print('19.汽水分离器进口蒸汽干度Xspi = ', x_spi, file=data)
      print(' 19.1.汽水分离器出口疏水压力puw = ', P_uw, file=data)
      print(' 19.2.汽水分离器出口疏水比焓huw = ', h_uw, file=data)
      print(' 第一级再热器', file=data)
      print('20.再热蒸汽进口压力prh1i = ', P_rh1i, file=data)
      print('21.再热蒸汽进口干度Xrh1i = ', x_rh1i, file=data)
      print(' 21.1.一级再热器进口蒸汽比焓hrh1i = ', h_rh1i, file=data)
      print('22.加热蒸汽进口压力prh1hs = ', P_rh1, file=data)
      print('23.加热蒸汽进口干度Xrh1hs = ', x_rh1, file=data)
      print(' 第二级再热器', file=data)
      print('24.再热蒸汽进口压力prh2i = ', P_rh2i, file=data)
      print('25.再热蒸汽进口温度Trh2i = ', T_rh2i, file=data)
      print('26.再热蒸汽出口压力prh2z = ', P_rh2z, file=data)
      print('27.再热蒸汽出口温度Trh2z = ', T_rh2z, file=data)
      print(' 27.1.二级再热器出口比焓hrh2z = ', h_rh2z, file=data)
      print(' 27.2.每级再热器平均焓升Δhrh = ', dh_rh, file=data)
      print(' 27.3.一级再热器出口蒸汽比焓hrh1z = ', h_rh1z, file=data)
      print(' 27.4.二级再热器进口蒸汽比焓hrh2i = ', h_rh2i, file=data)
      print('28.加热蒸汽进口压力prh2hs = ', P_rh2hs, file=data)
      print('29.加热蒸汽进口干度Xrh2hs = ', x_rh2hs, file=data)
      print(' 低压缸', file=data)
      print('30.进口蒸汽压力pli = ', P_li, file=data)
      print('31.进口蒸汽温度Tli = ', T_li, file=data)
      print('32.排汽压力plz = ', P_lz, file=data)
      print('33.排汽干度Xlz = ', x_lz, file=data)
      print(' 33.1.低压缸进口蒸汽比熵sli = ', s_li, file=data)
      print(' 33.2.低压缸进口蒸汽比焓hli = ', h_li, file=data)
      print(' 33.3.低压缸出口理想比焓hlzs = ', h_lzs, file=data)
      print(' 33.4.低压缸出口蒸汽比焓hlz = ', h_lz, file=data)
      print('34.回热级数Z', Z, file=data)
      print('35.低压给水加热器级数Zl = ', Z_l, file=data)
      print('36.高压给水加热器级数Zh = ', Z_h, file=data)
      print('37.第一次给水回热分配Δhfw = ', dh_fw, file=data)
      print(' 37.1.蒸汽发生器运行压力饱和水比焓hs = ', h_s, file=data)
      print(' 37.2.冷凝器出口凝结水比焓hcd = ', h_cd, file=data)
      print(' 37.3.每级加热器理论给水焓升Δhfwop = ', dh_fwop, file=data)
      print(' 37.4.最佳给水比焓hfwop = ', h_fwop, file=data)
      print(' 37.5.最佳给水温度Tfwop = ', T_fwop, file=data)
      print(' 37.6.实际给水温度Tfw = ', T_fw, file=data)
      print(' 37.7.实际给水比焓hfw = ', h_fw, file=data)
      print('38.高压加热器给水焓升Δhfwh = ', dh_fwh, file=data)
      print(' 38.1.除氧器运行压力pdea = ', P_dea, file=data)
      print(' 38.2.除氧器出口饱和水比焓hdeao = ', h_deao, file=data)
      print('39.除氧器及低压加热器给水焓升Δhfwl = ', dh_fwl, file=data)
      print(' 39.1.凝水泵出口给水压力pcwp = ', P_cwp, file=data)
      print(' 39.2.凝水泵出口给水比焓hcwp = ', h_cwp, file=data)
      print(' 39.3.凝水泵出口至除氧器出口的阻力压降Δpcws = ', dP_cws, file=data)
      print(' 39.4.每级低压加热器及除氧器的阻力压降Δpfi = ', dP_fi, file=data)
      print('40.低压加热器给水参数', file=data)
      print(' 第一级给水加热器参数', file=data)
      print(' 进口给水压力pfw1i = ', P_fw1i, file=data)
      print(' 进口给水比焓hfw1i = ', h_fw1i, file=data)
      print(' 进口给水温度Tfw1i = ', T_fw1i, file=data)
      print(' 出口给水压力pfw1o = ', P_fw1o, file=data)
      print(' 出口给水比焓hfw1o = ', h_fw1o, file=data)
      print(' 出口给水温度Tfw1o = ', T_fw1o, file=data)
      print(' 汽侧疏水温度Tro1k = ', T_ro1k, file=data)
      print(' 汽侧疏水比焓hro1k = ', h_ro1k, file=data)
      print(' 汽侧压力pro1k = ', P_ro1k, file=data)
      print(' 第二级给水加热器参数', file=data)
      print(' 进口给水压力pfw2i = ', P_fw2i, file=data)
      print(' 进口给水比焓hfw2i = ', h_fw2i, file=data)
      print(' 进口给水温度Tfw2i = ', T_fw2i, file=data)
      print(' 出口给水压力pfw2o = ', P_fw2o, file=data)
      print(' 出口给水比焓hfw2o = ', h_fw2o, file=data)
      print(' 出口给水温度Tfw2o = ', T_fw2o, file=data)
      print(' 汽侧疏水温度Tro2k = ', T_ro2k, file=data)
      print(' 汽侧疏水比焓hro2k = ', h_ro2k, file=data)
      print(' 汽侧压力pro2k = ', P_ro2k, file=data)
      print(' 第三级给水加热器参数', file=data)
      print(' 进口给水压力pfw3i = ', P_fw3i, file=data)
      print(' 进口给水比焓hfw3i = ', h_fw3i, file=data)
      print(' 进口给水温度Tfw3i = ', T_fw3i, file=data)
      print(' 出口给水压力pfw3o = ', P_fw3o, file=data)
      print(' 出口给水比焓hfw3o = ', h_fw3o, file=data)
      print(' 出口给水温度Tfw3o = ', T_fw3o, file=data)
      print(' 汽侧疏水温度Tro3k = ', T_ro3k, file=data)
      print(' 汽侧疏水比焓hro3k = ', h_ro3k, file=data)
      print(' 汽侧压力pro3k = ', P_ro3k, file=data)
      print(' 第四级给水加热器参数', file=data)
      print(' 进口给水压力pfw4i = ', P_fw4i, file=data)
      print(' 进口给水比焓hfw4i = ', h_fw4i, file=data)
      print(' 进口给水温度Tfw4i = ', T_fw4i, file=data)
      print(' 出口给水压力pfw4o = ', P_fw4o, file=data)
      print(' 出口给水比焓hfw4o = ', h_fw4o, file=data)
      print(' 出口给水温度Tfw4o = ', T_fw4o, file=data)
      print(' 汽侧疏水温度Tro4k = ', T_ro4k, file=data)
      print(' 汽侧疏水比焓hro4k = ', h_ro4k, file=data)
      print(' 汽侧压力pro4k = ', P_ro4k, file=data)
      print('除氧器', file=data)
      print('41.进口给水比焓hdeai = ', h_deai, file=data)
      print('42.出口给水比焓hdeao = ', h_deao, file=data)
      print('43.出口给水温度Tdea = ', T_dea, file=data)
      print('44.运行压力pdea = ', P_dea, file=data)
      print(' 44.1.给水泵出口压力pfwpo = ', P_fwpo, file=data)
      print(' 44.2.给水泵出口流体比焓hfwpo = ', h_fwpo, file=data)
      print(' 44.3.蒸汽发生器进口给水压力pfwi = ', P_fwi, file=data)
      print('45.高压加热器给水参数', file=data)
      print(' 第六级给水加热器参数', file=data)
      print(' 进口给水压力pfw6i = ', P_fw6i, file=data)
      print(' 进口给水比焓hfw6i = ', h_fw6i, file=data)
      print(' 进口给水温度Tfw6i = ', T_fw6i, file=data)
      print(' 出口给水压力pfw6o = ', P_fw6o, file=data)
      print(' 出口给水比焓hfw6o = ', h_fw6o, file=data)
      print(' 出口给水温度Tfw6o = ', T_fw6o, file=data)
      print(' 汽侧疏水温度Tro6k = ', T_ro6k, file=data)
      print(' 汽侧疏水比焓hro6k = ', h_ro6k, file=data)
      print(' 汽侧压力pro6k = ', P_ro6k, file=data)
      print(' 第七级给水加热器参数', file=data)
      print(' 进口给水压力pfw7i = ', P_fw7i, file=data)
      print(' 进口给水比焓hfw7i = ', h_fw7i, file=data)
      print(' 进口给水温度Tfw7i = ', T_fw7i, file=data)
      print(' 出口给水压力pfw7o = ', P_fw7o, file=data)
      print(' 出口给水比焓hfw7o = ', h_fw7o, file=data)
      print(' 出口给水温度Tfw7o = ', T_fw7o, file=data)
      print(' 汽侧疏水温度Tro7k = ', T_ro7k, file=data)
      print(' 汽侧疏水比焓hro7k = ', h_ro7k, file=data)
      print(' 汽侧压力pro7k = ', P_ro7k, file=data)
      print('46.高压缸抽汽', file=data)
      print(' 46.1.高压缸进口蒸汽比熵shi = ', s_hi, file=data)
      print(' 46.2.高压缸进口蒸汽比焓hhi = ', h_hi, file=data)
      print(' 第六级给水加热器抽汽参数', file=data)
      print(' 抽汽压力phes6 = ', P_hes6, file=data)
      print(' 抽汽干度Xhes6 = ', x_hes6, file=data)
      print(' 抽汽理想比焓hhes6s = ', h_hes6s, file=data)
      print(' 抽汽比焓hhes6 = ', h_hes6, file=data)
      print(' 第七级给水加热器抽汽参数', file=data)
      print(' 抽汽压力phes7 = ', P_hes7, file=data)
      print(' 抽汽干度Xhes7 = ', x_hes7, file=data)
      print(' 抽汽理想比焓hhes7s = ', h_hes7s, file=data)
      print(' 抽汽比焓hhes7 = ', h_hes7, file=data)
      print('47.低压缸抽汽', file=data)
      print(' 47.1.低压缸进口蒸汽比熵sli = ', s_li, file=data)
      print(' 47.2.低压缸进口蒸汽比焓hli = ', h_li, file=data)
      print(' 第一级给水加热器抽汽参数', file=data)
      print(' 抽汽压力ples1 = ', P_les1, file=data)
      print(' 抽汽干度Xles1 = ', x_les1, file=data)
      print(' 抽汽理想比焓hles1s = ', h_les1s, file=data)
      print(' 抽汽比焓hles1 = ', h_les1, file=data)
      print(' 第二级给水加热器抽汽参数', file=data)
      print(' 抽汽压力ples2 = ', P_les2, file=data)
      print(' 抽汽干度Xles2 = ', x_les2, file=data)
      print(' 抽汽理想比焓hles2s = ', h_les2s, file=data)
      print(' 抽汽比焓hles2 = ', h_les2, file=data)
      print(' 第三级给水加热器抽汽参数', file=data)
      print(' 抽汽压力ples3 = ', P_les3, file=data)
      print(' 抽汽干度Xles3 = ', x_les3, file=data)
      print(' 抽汽理想比焓hles3s = ', h_les3s, file=data)
      print(' 抽汽比焓hles3 = ', h_les3, file=data)
      print(' 第四级给水加热器抽汽参数', file=data)
      print(' 抽汽压力ples4 = ', P_les4, file=data)
      print(' 抽汽干度Xles4 = ', x_les4, file=data)
      print(' 抽汽理想比焓hles4s = ', h_les4s, file=data)
      print(' 抽汽比焓hles4 = ', h_les4, file=data)
      print('48.再热器抽汽', file=data)
      print(' 第一级再热器抽汽参数', file=data)
      print(' 加热蒸汽进口压力prh1 = ', P_rh1, file=data)
      print(' 加热蒸汽进口干度Xrh1 = ', x_rh1, file=data)
      print(' 加热蒸汽进口温度Trh1 = ', T_rh1, file=data)
      print(' 加热蒸汽进口比焓hrh1 = ', h_rh1, file=data)
      print(' 再热器疏水比焓hzs1 = ', h_zs1, file=data)
      print(' 第二级再热器抽汽参数', file=data)
      print(' 加热蒸汽进口压力prh2 = ', P_rh2, file=data)
      print(' 加热蒸汽进口干度Xrh2 = ', x_rh2, file=data)
      print(' 加热蒸汽进口温度Trh2 = ', T_rh2, file=data)
      print(' 加热蒸汽进口比焓hrh2 = ', h_rh2, file=data)
      print(' 再热器疏水比焓hzs2 = ', h_zs2, file=data)
      data.close()
      print("done")
      QMessageBox.information(self, "成功", "已成功输出结果文件！")

      # 展示迭代计算结果在主窗口的新的tab上
      fromParTab = 1
      global_var.set_value('fromParTab', fromParTab)
      self.outputSignal.emit()

class Exist(QDialog, Ui_exist):
   def __init__(self):
      super().__init__()  # 子窗口的实例化
      self.setupUi(self)
      self.buttonBox.accepted.connect(self.yesClicked)
      self.buttonBox.rejected.connect(self.noClicked)
   def yesClicked(self):
      global_var.set_value('overlay', 1)
      self.close()
   def noClicked(self):
      global_var.set_value('overlay', 2)
      self.close()