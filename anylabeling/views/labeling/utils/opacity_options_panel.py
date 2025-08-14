from PyQt5.QtWidgets import QWidget, QHBoxLayout, QToolButton, QSlider
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

from anylabeling.resources import resources


class OpacityOptionsPanel(QWidget):
    """
    A simplified panel for adjusting opacity of all shapes in real-time.
    
    This panel provides a horizontal slider and reset button for unified opacity
    control across all shape types (masks, polygons, rectangles, etc.).
    
    Attributes:
        opacity_slider (QSlider): The main opacity control slider.
        reset_btn (QToolButton): Button to reset opacity to default value.
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

        """)
        
        # Main layout
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
        
        # Connect reset button
        self.reset_btn.clicked.connect(self.reset_values)
    
    def get_opacity_values(self) -> tuple[int, int]:
        """
        Get the current opacity values from the slider.
        
        Returns:
            tuple[int, int]: A tuple containing (mask_opacity, fill_opacity) 
                           where both values are identical for unified opacity.
        
        Examples:
            >>> panel.get_opacity_values()
            (100, 100)
        """
        opacity = self.opacity_slider.value()
        return opacity, opacity  # Return same value for both mask and fill opacity
    
    def set_opacity_values(self, mask_opacity: int, fill_opacity: int) -> None:
        """
        Set the opacity values in the slider.
        
        Uses the minimum value between mask_opacity and fill_opacity to ensure
        consistent opacity across all shape types.
        
        Args:
            mask_opacity (int): The mask opacity value (0-255).
            fill_opacity (int): The fill opacity value (0-255).
        
        Examples:
            >>> panel.set_opacity_values(150, 200)
            >>> panel.opacity_slider.value()
            150
        """
        opacity = min(mask_opacity, fill_opacity)
        self.opacity_slider.setValue(opacity)
    
    def reset_values(self) -> None:
        """
        Reset the opacity slider to the default value.
        
        Sets the opacity slider to 100 (approximately 39% opacity) which
        provides good visibility while maintaining transparency.
        
        Examples:
            >>> panel.reset_values()
            >>> panel.opacity_slider.value()
            100
        """
        self.opacity_slider.setValue(100) 