from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6 import uic
import sys

# class UI(QWidget):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("untitled.ui",self)
#
# app = QApplication([])
# window = UI()
# window.show()
# sys.exit(app.exec())

import torch
print(torch.cuda.is_available())