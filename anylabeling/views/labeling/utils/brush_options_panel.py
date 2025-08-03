from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QSlider, QLabel
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
        
        # 메인 레이아웃을 세로로 변경
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        # 첫 번째 행: 브러시 크기
        brush_layout = QHBoxLayout()
        brush_layout.setSpacing(8)
        
        # brush_label = QLabel("브러시 크기:")
        # brush_label.setFixedWidth(80)
        # brush_layout.addWidget(brush_label)
        
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
        brush_layout.addWidget(self.slider)
        
        self.eraser_btn = QToolButton(self)
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setIcon(QIcon(":/images/images/eraser.svg"))
        self.eraser_btn.setIconSize(QSize(20, 20))
        self.eraser_btn.setFixedSize(36, 36)
        self.eraser_btn.setToolTip("지우개 모드")
        brush_layout.addWidget(self.eraser_btn)
        
        # 두 번째 행: 투명도 설정 버튼
        opacity_layout = QHBoxLayout()
        opacity_layout.setSpacing(8)
        
        opacity_label = QLabel("투명도:")
        opacity_label.setFixedWidth(80)
        opacity_layout.addWidget(opacity_label)
        
        # 투명도 설정 버튼만 남기고 슬라이더 제거
        self.opacity_btn = QToolButton(self)
        self.opacity_btn.setCheckable(True)
        # self.opacity_btn.setIcon(QIcon(":/images/images/opacity.png"))
        self.opacity_btn.setIconSize(QSize(20, 20))
        self.opacity_btn.setFixedSize(36, 36)
        self.opacity_btn.setToolTip("투명도 설정")
        opacity_layout.addWidget(self.opacity_btn)
        
        # 빈 공간 추가 (슬라이더 자리)
        opacity_layout.addStretch()
        
        # 레이아웃에 추가
        main_layout.addLayout(brush_layout)
        main_layout.addLayout(opacity_layout)
        
        self.setLayout(main_layout)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
        self.setMinimumHeight(120)
        self.setMaximumHeight(120)
