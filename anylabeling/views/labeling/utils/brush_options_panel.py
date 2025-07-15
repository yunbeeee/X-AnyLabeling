from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QSlider
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtSvg import QSvgRenderer

from anylabeling.resources import resources

class BrushOptionsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # self.slider = BrushSlider(Qt.Horizontal)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(10)
        self.slider.setValue(5)
        self.slider.setMinimumWidth(160)
        self.slider.setMaximumWidth(240)
        self.slider.setFixedHeight(40)
        layout.addWidget(self.slider)

        self.eraser_btn = QToolButton(self)
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setIcon(QIcon(":/images/images/eraser.svg"))
        self.eraser_btn.setIconSize(QSize(24, 24))
        self.eraser_btn.setFixedSize(40, 40)
        self.eraser_btn.setToolTip("지우개 모드")
        layout.addWidget(self.eraser_btn)

        self.setLayout(layout)
        self.setMinimumWidth(260)
        self.setMaximumWidth(400)
        self.setMinimumHeight(72)
        self.setMaximumHeight(72)
