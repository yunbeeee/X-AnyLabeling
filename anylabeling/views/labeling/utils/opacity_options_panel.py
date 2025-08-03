from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QSlider, QLabel, QSpinBox
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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        # 마스크 투명도 행
        mask_layout = QHBoxLayout()
        mask_layout.setSpacing(8)
        
        mask_label = QLabel("마스크:")
        mask_label.setFixedWidth(60)
        mask_layout.addWidget(mask_label)
        
        self.mask_slider = QSlider(Qt.Horizontal)
        self.mask_slider.setMinimum(0)
        self.mask_slider.setMaximum(255)
        self.mask_slider.setValue(100)
        self.mask_slider.setMinimumWidth(120)
        self.mask_slider.setMaximumWidth(180)
        self.mask_slider.setFixedHeight(32)
        mask_layout.addWidget(self.mask_slider)
        
        self.mask_spinbox = QSpinBox()
        self.mask_spinbox.setRange(0, 255)
        self.mask_spinbox.setValue(100)
        self.mask_spinbox.setFixedWidth(50)
        mask_layout.addWidget(self.mask_spinbox)
        
        # 채우기 투명도 행
        fill_layout = QHBoxLayout()
        fill_layout.setSpacing(8)
        
        fill_label = QLabel("채우기:")
        fill_label.setFixedWidth(60)
        fill_layout.addWidget(fill_label)
        
        self.fill_slider = QSlider(Qt.Horizontal)
        self.fill_slider.setMinimum(0)
        self.fill_slider.setMaximum(255)
        self.fill_slider.setValue(100)
        self.fill_slider.setMinimumWidth(120)
        self.fill_slider.setMaximumWidth(180)
        self.fill_slider.setFixedHeight(32)
        fill_layout.addWidget(self.fill_slider)
        
        self.fill_spinbox = QSpinBox()
        self.fill_spinbox.setRange(0, 255)
        self.fill_spinbox.setValue(100)
        self.fill_spinbox.setFixedWidth(50)
        fill_layout.addWidget(self.fill_spinbox)
        
        # 버튼 행
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.apply_btn = QToolButton(self)
        self.apply_btn.setIcon(QIcon(":/images/images/check.svg"))
        self.apply_btn.setIconSize(QSize(16, 16))
        self.apply_btn.setFixedSize(32, 32)
        self.apply_btn.setToolTip("적용")
        button_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QToolButton(self)
        # self.reset_btn.setIcon(QIcon(":/images/images/reset.svg"))
        self.reset_btn.setText("↺")
        self.reset_btn.setIconSize(QSize(16, 16))
        self.reset_btn.setFixedSize(32, 32)
        self.reset_btn.setToolTip("초기화")
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        # 레이아웃에 추가
        main_layout.addLayout(mask_layout)
        main_layout.addLayout(fill_layout)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        self.setMinimumHeight(140)
        self.setMaximumHeight(140)
        
        # 슬라이더와 스핀박스 연결
        self.mask_slider.valueChanged.connect(self.mask_spinbox.setValue)
        self.mask_spinbox.valueChanged.connect(self.mask_slider.setValue)
        
        self.fill_slider.valueChanged.connect(self.fill_spinbox.setValue)
        self.fill_spinbox.valueChanged.connect(self.fill_slider.setValue)
    
    def get_opacity_values(self):
        """투명도 값들을 튜플로 반환 (mask_opacity, fill_opacity)"""
        return self.mask_spinbox.value(), self.fill_spinbox.value()
    
    def set_opacity_values(self, mask_opacity, fill_opacity):
        """투명도 값들을 설정"""
        self.mask_spinbox.setValue(mask_opacity)
        self.fill_spinbox.setValue(fill_opacity)
    
    def reset_values(self):
        """기본값으로 리셋"""
        self.mask_spinbox.setValue(100)
        self.fill_spinbox.setValue(100) 