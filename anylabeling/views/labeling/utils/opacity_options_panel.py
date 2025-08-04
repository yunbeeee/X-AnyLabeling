from PyQt5.QtWidgets import QWidget, QHBoxLayout, QToolButton, QSlider
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

from anylabeling.resources import resources


class OpacityOptionsPanel(QWidget):
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
                padding: 4px;
                font-size: 14px;
                font-weight: bold;
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
            QSpinBox {
                background: white;
                border: 1px solid #d2d2d7;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }
        """)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(255)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setMinimumWidth(160)
        self.opacity_slider.setMaximumWidth(240)
        self.opacity_slider.setFixedHeight(32)
        self.opacity_slider.setStyleSheet("""
            QSlider {
                padding-left: 8px;
                padding-right: 8px;
            }
        """)
        main_layout.addWidget(self.opacity_slider)
        
        self.reset_btn = QToolButton(self)
        self.reset_btn.setIcon(QIcon(":/images/images/refresh.svg"))
        self.reset_btn.setIconSize(QSize(20, 20))
        self.reset_btn.setFixedSize(36, 36)
        self.reset_btn.setToolTip(self.tr("Reset all shapes opacity"))
        main_layout.addWidget(self.reset_btn)
        
        self.setLayout(main_layout)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        self.setMinimumHeight(80)
        self.setMaximumHeight(80)
        
        # 슬라이더와 초기화 버튼 연결
        self.reset_btn.clicked.connect(self.reset_values)
    
    def get_opacity_values(self):
        """Return opacity value"""
        opacity = self.opacity_slider.value()
        return opacity, opacity  # mask_opacity와 fill_opacity 모두 동일한 값 반환
    
    def set_opacity_values(self, mask_opacity, fill_opacity):
        """Set opacity values (use the lower value between the two)"""
        opacity = min(mask_opacity, fill_opacity)
        self.opacity_slider.setValue(opacity)
    
    def reset_values(self):
        """Reset to default value"""
        self.opacity_slider.setValue(100) 