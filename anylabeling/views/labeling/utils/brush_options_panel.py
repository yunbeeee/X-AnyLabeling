from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSlider, QToolButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

class BrushOptionsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup)  # 팝업처럼 동작, 포커스 잃으면 hide
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 브러시 크기 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(10)
        self.slider.setToolTip("브러시 크기")
        layout.addWidget(self.slider)

        # 지우개/브러시 토글 버튼
        self.eraser_btn = QToolButton()
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setIcon(QIcon(":/images/eraser.png"))  # 실제 경로로 수정
        self.eraser_btn.setToolTip("지우개 모드")
        layout.addWidget(self.eraser_btn)

        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.setFocus(Qt.PopupFocusReason)

    def focusOutEvent(self, event):
        self.hide()
        super().focusOutEvent(event)