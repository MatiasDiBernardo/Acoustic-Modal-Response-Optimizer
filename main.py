import sys
from PyQt5.QtWidgets import QApplication
from GUI import BROAcousticsGUI

app = QApplication(sys.argv)
ventana = BROAcousticsGUI()
ventana.show()
sys.exit(app.exec_())
