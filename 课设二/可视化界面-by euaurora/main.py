import os
import sys
from PyQt5.QtCore import pyqtSignal
import PyQt5
import global_var
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import parTab
import outTab
from Ui_Main import Ui_Main
from Ui_newPro import Ui_newPro
import Parameters

class MyMain(QMainWindow, Ui_Main):
    def __init__(self):
        super().__init__()  # 实例化
        self.setupUi(self)  # 加载窗口
        self.tabWidget_2.tabCloseRequested.connect(self.closeTab)           # tab关闭
    
        # 主界面按钮
        self.pushButton_openPro.clicked.connect(self.openPro)               # 打开项目
        self.pushButton_exit.clicked.connect(QApplication.instance().quit)  # 退出程序
        self.pushButton_newPro.clicked.connect(self.newPro)                 # 新建项目

        # 主菜单
        self.action_open.triggered.connect(self.openPro)    # 打开项目
        self.action_exit.triggered.connect(self.close)      # 退出程序
        self.action_new.triggered.connect(self.newPro)      # 新建项目
        self.action_openFile.triggered.connect(self.openFile)   # 打开文件
        
    # 打开项目
    def openPro(self):
        fileJson = None
        fileJson, ok = QFileDialog.getOpenFileName(self, "打开输入参数文件", "C:/", "json文件 (*.json)")
        if len(fileJson) != 0:
            mkpath = os.path.split(fileJson)[0]
            global_var.set_value('mkpath', mkpath)
            global_var.set_value('fileJson', fileJson)
            self.addParTab()
        
    # 新建项目
    def newPro(self):        
        self.newProWindow = newPro()
        self.newProWindow.show()

    # tab关闭
    def closeTab(self, index):
        self.tabWidget_2.removeTab(index)

    # 增加输入参数的tab
    def addParTab(self):
        self.parTabWin = parTab.parTabWindow()
        self.tabWidget_2.setCurrentIndex(self.tabWidget_2.addTab(self.parTabWin, "输入参数"))
        self.parTabWin.outputSignal.connect(lambda: MyMain.addOutTab(mainWindow))
    # 增加计算结果的tab
    def addOutTab(self):
        self.outTabWin = outTab.outTabWindow()
        self.tabWidget_2.setCurrentIndex(self.tabWidget_2.addTab(self.outTabWin, "输出文件"))
    
    # 打开文件显示在tab上
    def openFile(self):
        file = None
        file, ok = QFileDialog.getOpenFileName(self, "打开输入参数文件", "C:/", "json文件 (*.json);; 文本文件(*.txt);; Python文件(*.py) ")
        mkpath = os.path.split(file)[0]
        global_var.set_value('mkpath', mkpath)
        if len(file)!=0:
            if file.find('.json')!=-1:
                global_var.set_value('fileJson', file)
                self.addParTab()
            else:
                global_var.set_value('openFile', file)
                self.addOutTab()



# 新建项目窗口
class newPro(QDialog, Ui_newPro):
    def __init__(self):
        super().__init__()  # 子窗口的实例化
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.selectDic)
        self.buttonBox.accepted.connect(self.getFileInfo)
        self.buttonBox.rejected.connect(self.close)

    def selectDic(self):
        fileUrl = QFileDialog.getExistingDirectory(self, "New")
        self.lineEdit_2.setText(fileUrl)

    def getFileInfo(self):
        if(len(self.lineEdit.text())!=0):
            fileName = self.lineEdit.text()
        else:
            QMessageBox.warning(self, "警告", "请输入文件夹名称！")
        if(len(self.lineEdit_2.text())!=0):
            self.close()
            fileUrl = self.lineEdit_2.text()
             # 创建文件夹
            flag = 0
            mkpath = fileUrl + '/' + fileName
            global_var.set_value('mkpath', mkpath)
            exist = os.path.exists(mkpath)
            if not exist:
                os.makedirs(mkpath)
                flag = 1
            else:
                QMessageBox.warning(self, "警告", "文件夹已存在！")
            # 创建成功弹出参数窗口
            if(flag):
                self.parWindow = Parameters.ParWindow()
                self.parWindow.show()
                self.parWindow.parWindowClose.connect(lambda: MyMain.addParTab(mainWindow))
                
        else:
            QMessageBox.warning(self, "警告", "请选择路径！")
           
if __name__ == "__main__":
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    global_var._init()  # 全局变量处理器初始化
    mainWindow = MyMain()
    mainWindow.show()
    sys.exit(app.exec_())