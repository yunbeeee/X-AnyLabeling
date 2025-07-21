from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QSlider
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

from anylabeling.resources import resources

class BrushOptionsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background: white;
                border: 1px solid #ccc;
                border-radius: 8px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d2d2d7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0071e3;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #0071e3;
                border-radius: 2px;
            }
            QToolButton {
                background: #f5f5f7;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                padding: 8px;
            }
            QToolButton:hover {
                background: #e5e5e7;
                border-color: #0071e3;
            }
            QToolButton:pressed {
                background: #d5d5d7;
                border-color: #0071e3;
            }
            QToolButton:checked {
                background: #0071e3;
                border-color: #0071e3;
                color: white;
            }
            QToolButton:checked:hover {
                background: #005bbf;
                border-color: #005bbf;
            }
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
