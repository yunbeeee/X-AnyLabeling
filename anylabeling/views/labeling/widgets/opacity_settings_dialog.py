"""투명도 설정 다이얼로그"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

from anylabeling.views.labeling.utils.qt import new_icon_path


class OpacitySettingsDialog(QtWidgets.QDialog):
    def __init__(self, mask_opacity=100, fill_opacity=100, parent=None):
        super().__init__(parent)

        self._mask_opacity = mask_opacity
        self._fill_opacity = fill_opacity

        self.setWindowTitle(self.tr("투명도 설정"))
        self.setModal(True)
        self.setFixedSize(380, 200)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )

        # Apply macOS style
        self.setStyleSheet(
            f"""
                QDialog {{
                    background-color: #f5f5f7;
                    border-radius: 10px;
                }}
                QLabel {{
                    color: #1d1d1f;
                    font-size: 13px;
                }}
                QSlider {{
                    height: 28px;
                }}
                QSlider::groove:horizontal {{
                    height: 4px;
                    background: #d2d2d7;
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: #0071e3;
                    border: none;
                    width: 16px;
                    height: 16px;
                    margin: -6px 0;
                    border-radius: 8px;
                }}
                QSlider::sub-page:horizontal {{
                    background: #0071e3;
                    border-radius: 2px;
                }}
                QSpinBox {{
                    padding: 5px 8px;
                    background: white;
                    border: 1px solid #d2d2d7;
                    border-radius: 6px;
                    min-height: 24px;
                }}
            """
        )

        # Create layout with proper spacing
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Mask opacity controls
        mask_layout = QtWidgets.QHBoxLayout()
        mask_layout.setSpacing(10)
        self.mask_label = QtWidgets.QLabel(self.tr("마스크 투명도:"))
        self.mask_label.setMinimumWidth(100)
        mask_layout.addWidget(self.mask_label)

        self.mask_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mask_slider.setMinimum(0)
        self.mask_slider.setMaximum(255)
        self.mask_slider.setValue(self._mask_opacity)
        self.mask_slider.setTickInterval(25)
        mask_layout.addWidget(self.mask_slider)

        self.mask_spinbox = QtWidgets.QSpinBox()
        self.mask_spinbox.setRange(0, 255)
        self.mask_spinbox.setValue(self._mask_opacity)
        self.mask_spinbox.setFixedWidth(68)
        mask_layout.addWidget(self.mask_spinbox)

        # Fill opacity controls
        fill_layout = QtWidgets.QHBoxLayout()
        fill_layout.setSpacing(10)
        self.fill_label = QtWidgets.QLabel(self.tr("채우기 투명도:"))
        self.fill_label.setMinimumWidth(100)
        fill_layout.addWidget(self.fill_label)

        self.fill_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fill_slider.setMinimum(0)
        self.fill_slider.setMaximum(255)
        self.fill_slider.setValue(self._fill_opacity)
        self.fill_slider.setTickInterval(25)
        fill_layout.addWidget(self.fill_slider)

        self.fill_spinbox = QtWidgets.QSpinBox()
        self.fill_spinbox.setRange(0, 255)
        self.fill_spinbox.setValue(self._fill_opacity)
        self.fill_spinbox.setFixedWidth(68)
        fill_layout.addWidget(self.fill_spinbox)

        # Buttons layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)

        # Reset button
        self.reset_button = QtWidgets.QPushButton(self.tr("초기화"))
        self.reset_button.setFixedSize(100, 32)
        self.reset_button.clicked.connect(self.reset_values)
        self.reset_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
        """
        )

        # OK and Cancel buttons
        ok_button = QtWidgets.QPushButton(self.tr("확인"))
        ok_button.setFixedSize(100, 32)
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0071e3;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0077ED;
            }
            QPushButton:pressed {
                background-color: #0068D0;
            }
        """
        )

        cancel_button = QtWidgets.QPushButton(self.tr("취소"))
        cancel_button.setFixedSize(100, 32)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
        """
        )

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        # Add all layouts to the main layout
        layout.addLayout(mask_layout)
        layout.addLayout(fill_layout)
        layout.addStretch(1)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals for slider and spinbox synchronization
        self.mask_slider.valueChanged.connect(self.mask_spinbox.setValue)
        self.mask_spinbox.valueChanged.connect(self.mask_slider.setValue)

        self.fill_slider.valueChanged.connect(self.fill_spinbox.setValue)
        self.fill_spinbox.valueChanged.connect(self.fill_slider.setValue)

        self.move_to_center()

    def move_to_center(self):
        """Move dialog to center of the screen"""
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def reset_values(self):
        """Reset sliders to default values"""
        self.mask_slider.setValue(100)
        self.fill_slider.setValue(100)

    def get_settings(self):
        return {
            "mask_opacity": self.mask_spinbox.value(),
            "fill_opacity": self.fill_spinbox.value(),
        }
    
    def get_opacity_values(self):
        """투명도 값들을 튜플로 반환 (mask_opacity, fill_opacity)"""
        return self.mask_spinbox.value(), self.fill_spinbox.value()