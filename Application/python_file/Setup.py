# don't change anything, Used for setup UI for all pages
import sys
from PySide2 import QtCore
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication,QVBoxLayout
from PySide2.QtCore import QFile, QCoreApplication, QIODevice


class Setup_urls:
    def __init__(self,url):
        self.ui_file_name = url
        self.ui_file = QFile(self.ui_file_name)

        if not self.ui_file.open(QIODevice.ReadOnly):                                                         # Error Occure if file not open
            print("Cannot open {}: {}".format(self.ui_file_name, ui_file.errorString()))
            sys.exit(-1)

        self.window=QUiLoader().load(self.ui_file)
        self.ui_file.close()
if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    Object1 = Setup_urls("../UI/sheet_design.ui")
    Object1.window.show()
    sys.exit(app.exec_())
