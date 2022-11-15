import os
from posixpath import split 
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from Ui_outTab import Ui_outTab
from Ui_exist import Ui_exist
import global_var

class outTabWindow(QWidget, Ui_outTab):
    def __init__(self):
        super().__init__()  # 子窗口的实例化
        self.setupUi(self)
        self.loadFile()
        self.pushButton_py.clicked.connect(self.generatePy)

    def loadFile(self):
        if (global_var.get_value('fromParTab')):
            fileJson = global_var.get_value('fileJson')
            fileName = fileJson.split(".json")[0] + '_' + 'interation.txt'
            file = open(fileName, 'r', encoding='utf-8')
            fileContent = file.read()
            file.close()
            self.plainTextEdit.setPlainText(fileContent)

        else:
            fileName = global_var.get_value('openFile')
            file = open(fileName, 'r', encoding='utf-8')
            fileContent = file.read()
            file.close()
            self.plainTextEdit.setPlainText(fileContent)
    def generatePy(self):
        QMessageBox.information(self, "提示", "此功能静待有缘人……")