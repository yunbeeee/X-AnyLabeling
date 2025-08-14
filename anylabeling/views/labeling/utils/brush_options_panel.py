from PyQt5.QtWidgets import QWidget, QHBoxLayout, QToolButton, QSlider
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

from anylabeling.resources import resources


class BrushOptionsPanel(QWidget):
    """
    A panel for brush size control and eraser mode toggle.
    
    This panel provides a horizontal slider for brush size adjustment and
    an eraser button for toggling between drawing and erasing modes.
    
    Attributes:
        slider (QSlider): The brush size control slider.
        eraser_btn (QToolButton): Button to toggle eraser mode.
    """
    
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
                margin: 0 8px;
                padding: 0 8px;
            }
            QSlider::handle:horizontal {
                background: #0071e3;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px -20px;
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
            QLabel {
                color: #1d1d1f;
                font-size: 12px;
                font-weight: 500;
            }
        """)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(250)
        self.slider.setValue(120)
        self.slider.setMinimumWidth(160)
        self.slider.setMaximumWidth(240)
        self.slider.setFixedHeight(32)
        self.slider.setStyleSheet("""
            QSlider {
                padding-left: 8px;
                padding-right: 8px;
            }
        """)
        main_layout.addWidget(self.slider)
        
        self.eraser_btn = QToolButton(self)
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setIcon(QIcon(":/images/images/eraser.svg"))
        self.eraser_btn.setIconSize(QSize(20, 20))
        self.eraser_btn.setFixedSize(36, 36)
        self.eraser_btn.setToolTip(self.tr("Eraser Mode"))
        main_layout.addWidget(self.eraser_btn)
        
        self.setLayout(main_layout)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        self.setMinimumHeight(80)
        self.setMaximumHeight(80)
