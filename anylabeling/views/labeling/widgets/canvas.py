"""This module defines Canvas widget - the core component for drawing image labels"""

import math
import numpy as np
import cv2
from copy import deepcopy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QWheelEvent, QImage, QColor, QPolygonF

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.views.labeling.utils.colormap import label_colormap
from anylabeling.views.labeling.utils.mask_utils import apply_brush_to_mask
from anylabeling.views.labeling.utils.mask_utils import mask_to_polygon
from anylabeling.views.labeling.utils.mask_utils import polygon_to_mask

from .. import utils
from ..shape import Shape

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0
LARGE_ROTATION_INCREMENT = 0.1
SMALL_ROTATION_INCREMENT = 0.01

LABEL_COLORMAP = label_colormap()


class Canvas(
    QtWidgets.QWidget
):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Canvas widget to handle label drawing"""

    zoom_request = QtCore.pyqtSignal(int, QtCore.QPoint)
    scroll_request = QtCore.pyqtSignal(float, int, int)
    # [Feature] support for automatically switching to editing mode
    # when the cursor moves over an object
    mode_changed = QtCore.pyqtSignal()
    new_shape = QtCore.pyqtSignal()
    show_shape = QtCore.pyqtSignal(int, int, QtCore.QPointF)
    selection_changed = QtCore.pyqtSignal(list)
    shape_moved = QtCore.pyqtSignal()
    shape_rotated = QtCore.pyqtSignal()
    drawing_polygon = QtCore.pyqtSignal(bool)
    vertex_selected = QtCore.pyqtSignal(bool)
    auto_labeling_marks_updated = QtCore.pyqtSignal(list)
    brush_mode_changed = QtCore.pyqtSignal(bool)  # 브러시 모드 on/off 알림
    eraser_mode_changed = QtCore.pyqtSignal(bool)  # 지우개 모드 on/off 알림

    CREATE, EDIT = 0, 1

    # polygon, rectangle, rotation, line, or point
    _create_mode = "mask"

    _fill_drawing = True

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                f"Unexpected value for double_click event: {self.double_click}"
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self.wheel_rectangle_editing = kwargs.pop(
            "wheel_rectangle_editing", {}
        )
        self.enable_wheel_rectangle_editing = self.wheel_rectangle_editing.get(
            "enable", False
        )
        self.rect_adjust_step = self.wheel_rectangle_editing.get(
            "adjust_step", 2.0
        )
        self.rect_scale_step = self.wheel_rectangle_editing.get(
            "scale_step", 0.05
        )
        self.parent = kwargs.pop("parent")
        super().__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.is_auto_labeling = False
        self.is_move_editing = False
        self.auto_labeling_mode: AutoLabelingMode = None
        self.shapes = []
        self.shapes_backups = []
        self.current = None
        self.selected_shapes = []  # save the selected shapes here
        self.selected_shapes_copy = []
        # self.line represents:
        #   - create_mode == 'polygon': edge from last point to current
        #   - create_mode == 'rectangle': diagonal line of the rectangle
        #   - create_mode == 'line': the line
        #   - create_mode == 'point': the point
        self.line = Shape()
        self.prev_point = QtCore.QPoint()
        self.prev_pan_point = QtCore.QPoint()
        self.prev_move_point = QtCore.QPoint()
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hide_backround = False
        self.hide_backround = False
        self.h_hape = None
        self.prev_h_shape = None
        self.h_vertex = None
        self.prev_h_vertex = None
        self.h_edge = None
        self.prev_h_edge = None
        self.moving_shape = False
        self.rotating_shape = False
        self.snapping = True
        self.h_shape_is_selected = False
        self.h_shape_is_hovered = None
        self.allowed_oop_shape_types = ["rotation"]
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.show_groups = False
        self.show_texts = True
        self.show_labels = True
        self.show_scores = True
        self.show_degrees = False
        self.show_linking = True

        # Set cross line options.
        self.cross_line_show = True
        self.cross_line_width = 2.0
        self.cross_line_color = "#00FF00"
        self.cross_line_opacity = 0.5

        self.is_loading = False
        self.loading_text = self.tr("Loading...")
        self.loading_angle = 0

        # Brush mode init
        self.brush_radius = 10
        self.is_brush_mode = False
        self.brush_modified = False
        self._mask_qimage_cache = {}
        self._mask_overlay_cache = {}
        self.eraser_mode = False  # ← 브러시/지우개 모드 상태 변수 초기화
        self._brush_target_shape = None  
        self._prev_brush_pos = None  
        
        # 성능 최적화 변수들
        self._brush_update_timer = QtCore.QTimer()
        self._brush_update_timer.setSingleShot(True)
        self._brush_update_timer.timeout.connect(self._delayed_brush_update)
        self._brush_stroke_batch = []  # 배치 처리용

    def set_loading(self, is_loading: bool, loading_text: str = None):
        """Set loading state"""
        self.is_loading = is_loading
        if loading_text:
            self.loading_text = loading_text
        self.update()

    def set_auto_labeling_mode(self, mode: AutoLabelingMode):
        """Set auto labeling mode"""
        if mode == AutoLabelingMode.NONE:
            self.is_auto_labeling = False
            self.auto_labeling_mode = mode
        else:
            self.is_auto_labeling = True
            self.auto_labeling_mode = mode
            self.create_mode = mode.shape_type
            self.parent.toggle_draw_mode(
                False, mode.shape_type, disable_auto_labeling=False
            )

    def fill_drawing(self):
        """Get option to fill shapes by color"""
        return self._fill_drawing

    def set_fill_drawing(self, value):
        """Set shape filling option"""
        self._fill_drawing = value

    @property
    def create_mode(self):
        """Create mode for canvas - Modes: polygon, rectangle, rotation, circle,..."""
        return self._create_mode

    @create_mode.setter
    def create_mode(self, value):
        """Set create mode for canvas"""
        if value not in [
            "polygon",
            "rectangle",
            "rotation",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            raise ValueError(f"Unsupported create_mode: {value}")
        self._create_mode = value

    def store_shapes(self):
        """Store shapes for restoring later (Undo feature)"""
        shapes_backup = []
        for shape in self.shapes:
            shapes_backup.append(shape.copy())
        if len(self.shapes_backups) > self.num_backups:
            self.shapes_backups = self.shapes_backups[-self.num_backups - 1 :]
        self.shapes_backups.append(shapes_backup)

    def store_moving_shape(self):
        """Store a moving shape"""
        if self.moving_shape:
            moving_shapes = (
                [self.h_hape] + self.selected_shapes
                if self.h_hape and self.h_hape not in self.selected_shapes
                else self.selected_shapes.copy()
            )
            for shape in moving_shapes:
                if shape in self.shapes:
                    index = self.shapes.index(shape)
                    if (
                        len(self.shapes_backups) > 0
                        and index < len(self.shapes_backups[-1])
                        and self.shapes_backups[-1][index].points
                        != self.shapes[index].points
                    ):
                        self.store_shapes()
                        self.shape_moved.emit()
                        break

            self.moving_shape = False

    @property
    def is_shape_restorable(self):
        """Check if shape can be restored from backup"""
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapes_backups) < 2:
            return False
        return True

    def restore_shape(self):
        """Restore/Undo a shape"""
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::load_shapes and our own Canvas::load_shapes function.
        if not self.is_shape_restorable:
            return
        self.shapes_backups.pop()  # latest

        # The application will eventually call Canvas.load_shapes which will
        # push this right back onto the stack.
        shapes_backup = self.shapes_backups.pop()
        self.shapes = shapes_backup
        self.selected_shapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, _):
        """Mouse enter event"""
        self.override_cursor(self._cursor)

    def leaveEvent(self, _):
        """Mouse leave event"""
        self.store_moving_shape()
        self.un_highlight()
        self.restore_cursor()

    def focusOutEvent(self, _):
        """Window out of focus event"""
        self.restore_cursor()

    def is_visible(self, shape):
        """Check if a shape is visible"""
        return self.visible.get(shape, True)

    def drawing(self):
        """Check if user is drawing (mode==CREATE)"""
        return self.mode == self.CREATE

    def editing(self):
        """Check if user is editing (mode==EDIT)"""
        return self.mode == self.EDIT

    def set_auto_labeling(self, value=True):
        """Set auto labeling mode"""
        self.is_auto_labeling = value
        if self.auto_labeling_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
            self.parent.toggle_draw_mode(
                True, "rectangle", disable_auto_labeling=True
            )

    def get_mode(self):
        """Get current mode"""
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            return self.tr("Auto Labeling")
        if self.mode == self.CREATE:
            return self.tr("Drawing")
        elif self.mode == self.EDIT:
            return self.tr("Editing")
        else:
            return self.tr("Unknown")

    def set_editing(self, value=True):
        """Set editing mode. Editing is set to False, user is drawing"""
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.un_highlight()
            self.deselect_shape()
            self.is_move_editing = False

    def un_highlight(self):
        """Unhighlight shape/vertex/edge"""
        if self.h_hape:
            self.h_hape.highlight_clear()
            self.update()
        self.prev_h_shape = self.h_hape
        self.prev_h_vertex = self.h_vertex
        self.prev_h_edge = self.h_edge
        self.h_hape = self.h_vertex = self.h_edge = None

    def selected_vertex(self):
        """Check if selected a vertex"""
        return self.h_vertex is not None

    def selected_edge(self):
        """Check if selected an edge"""
        return self.h_edge is not None

    # QT Overload
    def mouseMoveEvent(self, ev):  # noqa: C901
        """Update line with last point and current coordinates"""
        if self.is_loading:
            return
        try:
            canvas_pos = ev.localPos()  # 또는 ev.pos()
            image_pos = self.transform_pos(canvas_pos)  # 반드시 이미지 좌표계로 변환!
        except AttributeError:
            return

        self.prev_move_point = image_pos
        self.repaint()

        # Polygon drawing.
        if self.drawing():
            line_color = utils.hex_to_rgb(self.cross_line_color)
            self.line.line_color = QtGui.QColor(*line_color)
            self.line.shape_type = self.create_mode

            if not self.current:
                self.override_cursor(CURSOR_DRAW)
                return

            if self.create_mode == "rectangle":
                shape_width = int(abs(self.current[0].x() - image_pos.x()))
                shape_height = int(abs(self.current[0].y() - image_pos.y()))
                self.show_shape.emit(shape_width, shape_height, image_pos)

            color = QtGui.QColor(0, 0, 255)
            if (
                self.out_off_pixmap(image_pos)
                and self.create_mode not in self.allowed_oop_shape_types
            ):
                # Don't allow the user to draw outside the pixmap, except for rotation.
                # Project the point to the pixmap's edges.
                image_pos = self.intersection_point(self.current[-1], image_pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.create_mode == "polygon"
                and self.close_enough(image_pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                image_pos = self.current[0]
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            elif (
                self.create_mode == "rotation"
                and len(self.current) > 0
                and self.close_enough(image_pos, self.current[0])
            ):
                image_pos = self.current[0]
                color = self.current.line_color
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            else:
                self.override_cursor(CURSOR_DRAW)
            if self.create_mode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = image_pos
            elif self.create_mode == "rectangle":
                self.line.points = [self.current[0], image_pos]
                self.line.close()
            elif self.create_mode == "rotation":
                self.line[1] = image_pos
                self.line.line_color = color
            elif self.create_mode == "circle":
                self.line.points = [self.current[0], image_pos]
                self.line.shape_type = "circle"
            elif self.create_mode == "line":
                self.line.points = [self.current[0], image_pos]
                self.line.close()
            elif self.create_mode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.repaint()
            self.current.highlight_clear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selected_shapes_copy and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes_copy, image_pos)
                self.repaint()
            elif self.selected_shapes:
                self.selected_shapes_copy = [
                    s.copy() for s in self.selected_shapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.editing() and self.is_brush_mode and self._brush_target_shape is not None:
                # Ctrl 키가 눌려있으면 지우기 모드, 아니면 그리기 모드
                # UI 버튼 상태와 Ctrl 키 상태를 모두 고려
                shift_pressed = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
                add = not (self.eraser_mode or shift_pressed)
                curr = image_pos
                prev = getattr(self, '_prev_brush_pos', None)
                
                # 브러시 크기를 화면 좌표로 일정하게 유지 (줌 레벨 적용)
                image_brush_radius = max(1, int(self.brush_radius / self.scale))
                
                # 성능 최적화: 브러시 스트로크 배치 처리
                if prev is not None:
                    dist = ((curr.x() - prev.x()) ** 2 + (curr.y() - prev.y()) ** 2) ** 0.5
                    # 거리가 작으면 중간점 보간 생략 (성능 향상)
                    if dist > image_brush_radius * 0.5:
                        step_size = max(2, image_brush_radius // 3)  # 스텝 크기 증가로 호출 횟수 감소
                        steps = max(1, int(dist // step_size))
                        for i in range(steps + 1):
                            t = i / steps
                            x = prev.x() * (1 - t) + curr.x() * t
                            y = prev.y() * (1 - t) + curr.y() * t
                            self.edit_mask_with_brush(self._brush_target_shape, QtCore.QPointF(x, y), radius=image_brush_radius, add=add)
                    else:
                        # 거리가 짧으면 현재 점만 처리
                        self.edit_mask_with_brush(self._brush_target_shape, curr, radius=image_brush_radius, add=add)
                else:
                    self.edit_mask_with_brush(self._brush_target_shape, curr, radius=image_brush_radius, add=add)
                
                # 지연 업데이트로 렌더링 최적화
                self._brush_update_timer.stop()
                self._brush_update_timer.start(8)  # 8ms 지연 (약 120fps)
                self._prev_brush_pos = curr
                return
            if self.selected_vertex():
                self.is_move_editing = False
                try:
                    self.bounded_move_vertex(image_pos)
                    self.repaint()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_hape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, image_pos)
            elif self.selected_shapes and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes, image_pos)
                self.repaint()
                self.moving_shape = True
                if self.selected_shapes[-1].shape_type == "rectangle":
                    p1 = self.selected_shapes[-1][0]
                    p2 = self.selected_shapes[-1][2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, image_pos)
            else:
                # 이미지 팬 기능 비활성화 (의도치 않은 이미지 이동 방지)
                # if (
                #     self.pixmap
                #     and self.pixmap.width()
                #     and self.pixmap.height()
                # ):
                #     self.override_cursor(CURSOR_MOVE)
                #     delta = image_pos - self.prev_pan_point
                #     self.scroll_request.emit(
                #         delta.x() / (self.pixmap.width() * self.scale),
                #         Qt.Horizontal,
                #         1,
                #     )
                #     self.scroll_request.emit(
                #         delta.y() / (self.pixmap.height() * self.scale),
                #         Qt.Vertical,
                #     )
                #     self.repaint()
                pass
            return

        if self.editing() and self.is_move_editing:
            self.override_cursor(CURSOR_MOVE)
            if self.selected_vertex():
                try:
                    self.bounded_move_vertex(image_pos)
                    self.repaint()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_hape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, image_pos)
            else:
                self.is_move_editing = False

            return

        self.show_shape.emit(-1, -1, image_pos)

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.is_visible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearest_vertex(image_pos, self.epsilon / self.scale)
            index_edge = shape.nearest_edge(image_pos, self.epsilon / self.scale)
            if index is not None:
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex = index
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                shape.highlight_vertex(index, shape.MOVE_VERTEX)
                self.override_cursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click & drag to move point of shape '%s'")
                    % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            if index_edge is not None and shape.can_add_point():
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge = index_edge
                self.override_cursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click to create point of shape '%s'")
                    % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            if len(shape.points) > 1 and shape.contains_point(image_pos):
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                if shape.group_id and shape.shape_type == "rectangle":
                    tooltip_text = "Click & drag to move shape '{label} {group_id}'".format(
                        label=shape.label, group_id=shape.group_id
                    )
                    self.setToolTip(self.tr(tooltip_text))
                else:
                    self.setToolTip(
                        self.tr("Click & drag to move shape '%s'")
                        % shape.label
                    )
                self.setStatusTip(self.toolTip())
                self.override_cursor(CURSOR_GRAB)
                # [Feature] Automatically highlight shape when the mouse is moved inside it
                if self.h_shape_is_hovered:
                    group_mode = (
                        int(ev.modifiers()) == QtCore.Qt.ControlModifier
                    )
                    self.select_shape_point(
                        image_pos, multiple_selection_mode=group_mode
                    )
                self.update()

                if shape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, image_pos)
                break
        else:  # Nothing found, clear highlights, reset state.
            self.un_highlight()
            self.override_cursor(CURSOR_DEFAULT)
        self.vertex_selected.emit(self.h_vertex is not None)

    def add_point_to_edge(self):
        """Add a point to current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_edge
        point = self.prev_move_point
        if shape is None or index is None or point is None:
            return
        shape.insert_point(index, point)
        shape.highlight_vertex(index, shape.MOVE_VERTEX)
        self.h_hape = shape
        self.h_vertex = index
        self.h_edge = None
        self.moving_shape = True

    def remove_selected_point(self):
        """Remove a point from current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_vertex
        if shape is None or index is None:
            return
        shape.remove_point(index)
        shape.highlight_clear()
        self.h_hape = shape
        self.prev_h_vertex = None
        self.moving_shape = True  # Save changes

    # QT Overload
    def mousePressEvent(self, ev):  # noqa: C901
        """Mouse press event"""
        if self.is_loading:
            return
        canvas_pos = ev.localPos() 
        image_pos = self.transform_pos(canvas_pos) 
        x, y = int(image_pos.x()), int(image_pos.y())

        if ev.button() == QtCore.Qt.LeftButton:

            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.create_mode == "polygon":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode in ["circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.create_mode == "rectangle":
                        if self.current.reach_max_points() is False:
                            init_pos = self.current[0]
                            min_x = init_pos.x()
                            min_y = init_pos.y()
                            target_pos = self.line[1]
                            max_x = target_pos.x()
                            max_y = target_pos.y()
                            self.current.add_point(
                                QtCore.QPointF(max_x, min_y)
                            )
                            self.current.add_point(target_pos)
                            self.current.add_point(
                                QtCore.QPointF(min_x, max_y)
                            )
                            self.finalise()
                    elif self.create_mode == "rotation":
                        initPos = self.current[0]
                        minX = initPos.x()
                        minY = initPos.y()
                        targetPos = self.line[1]
                        maxX = targetPos.x()
                        maxY = targetPos.y()
                        self.current.add_point(QtCore.QPointF(maxX, minY))
                        self.current.add_point(targetPos)
                        self.current.add_point(QtCore.QPointF(minX, maxY))
                        self.current.add_point(initPos)
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode == "linestrip":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    # [Feature] support for automatically switching to editing mode
                    # when the cursor moves over an object
                    if (
                        self.create_mode
                        in ["rectangle", "rotation", "circle", "line", "point"]
                        and not self.is_auto_labeling
                    ):
                        self.mode_changed.emit()
                elif not self.out_off_pixmap(image_pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(image_pos)
                    if self.create_mode == "point":
                        self.finalise()
                    else:
                        if self.create_mode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [image_pos, image_pos]
                        self.set_hiding()
                        self.drawing_polygon.emit(True)
                        self.update()
                elif (
                    self.out_off_pixmap(image_pos)
                    and self.create_mode in self.allowed_oop_shape_types
                ):
                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(image_pos)
                    self.line.points = [image_pos, image_pos]
                    self.set_hiding()
                    self.drawing_polygon.emit(True)
                    self.update()
            elif self.editing():                    
                if self.is_brush_mode:
                    if self.out_off_pixmap(image_pos):
                        self.set_brush_mode(False)
                        return
                    
                    # 브러시 타겟 설정
                    brush_target = None
                    
                    # 1. 기존 타겟이 유효하고 클릭 위치에 있으면 계속 사용 (연속성 보장)
                    if (hasattr(self, '_brush_target_shape') and 
                        self._brush_target_shape is not None and 
                        self._brush_target_shape in self.shapes and 
                        self.is_visible(self._brush_target_shape) and
                        self._brush_target_shape.contains_point(image_pos)):
                        brush_target = self._brush_target_shape
                    
                    # 2. 기존 타겟이 유효하지 않으면 선택된 도형 우선 확인 (사용자 의도 반영)
                    elif self.selected_shapes:
                        first_selected = self.selected_shapes[0]
                        if (first_selected.is_mask() or first_selected.shape_type in ["polygon", "rectangle", "rotation"]) and first_selected.contains_point(image_pos):
                            brush_target = first_selected
                    
                    # 3. 그래도 없으면 클릭한 위치의 브러시 가능한 도형 찾기
                    if brush_target is None:
                        for shape in reversed(self.shapes):
                            if not self.is_visible(shape):
                                continue
                            if (shape.is_mask() or shape.shape_type in ["polygon", "rectangle", "rotation"]) and shape.contains_point(image_pos):
                                brush_target = shape
                                break
                    
                    # 4. 브러시 타겟 설정 및 선택 상태 동기화
                    if brush_target != getattr(self, '_brush_target_shape', None):
                        # 이전 타겟의 변경사항 저장 (중요!)
                        if (hasattr(self, '_brush_target_shape') and 
                            self._brush_target_shape is not None and 
                            self.brush_modified):
                            self.store_shapes()
                            self.shape_moved.emit()  # undo 활성화
                            self.brush_modified = False
                        
                        self._brush_target_shape = brush_target
                        # 타겟이 변경되면 선택 상태도 동기화
                        if self._brush_target_shape is not None and self._brush_target_shape not in self.selected_shapes:
                            self.selected_shapes = [self._brush_target_shape]
                            self.selection_changed.emit(self.selected_shapes)
                    
                    self._prev_brush_pos = None  # 브러시 드래그 시작점 초기화

                    # 브러시 타겟이 있으면 즉시 브러시 적용
                    if self._brush_target_shape is not None:
                        # UI 버튼 상태와 Ctrl 키 상태를 모두 고려
                        ctrl_pressed = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
                        add = not (self.eraser_mode or ctrl_pressed)  # Ctrl for erasing
                        
                        # 브러시 크기를 화면 좌표로 일정하게 유지 (줌 레벨 적용)
                        image_brush_radius = max(1, int(self.brush_radius / self.scale))
                        self.edit_mask_with_brush(self._brush_target_shape, image_pos, radius=image_brush_radius, add=add)
                        self.brush_modified = True
                        
                        # 브러시 모드에서는 도형 선택 로직을 건너뛰고 바로 return
                        self.prev_pan_point = ev.localPos()
                        self.repaint()
                        return

                if self.selected_edge():
                    self.add_point_to_edge()
                elif (
                    self.selected_vertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                    and self.h_hape.shape_type
                    not in ["rectangle", "rotation", "line"]
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.remove_selected_point()


                if self.selected_vertex():
                    self.is_move_editing = not self.is_move_editing
                    if self.is_move_editing:
                        self.override_cursor(CURSOR_MOVE)
                    else:
                        self.override_cursor(CURSOR_POINT)

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.select_shape_point(
                    image_pos, multiple_selection_mode=group_mode
                )
                self.prev_point = image_pos

        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selected_shapes or (
                self.h_hape is not None
                and self.h_hape not in self.selected_shapes
            ):
                self.select_shape_point(
                    image_pos, multiple_selection_mode=group_mode
                )
                self.repaint()
            self.prev_point = image_pos
        

    # QT Overload
    def mouseReleaseEvent(self, ev):
        """Mouse release event"""
        if self.is_loading:
            return

        if self.brush_modified:
            self.store_shapes()
            # 브러시 타이머 정리 및 최종 업데이트
            self._brush_update_timer.stop()
            self.update()  # 최종 고품질 렌더링
            # 브러시 수정 완료 시 selection_changed 시그널 발생
            if self.selected_shapes:
                self.selection_changed.emit(self.selected_shapes)
            # 브러시 수정 완료 시 shape_moved 시그널 발생 (undo 활성화)
            self.shape_moved.emit()
            self.brush_modified = False

        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selected_shapes_copy) > 0]
            self.restore_cursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selected_shapes_copy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selected_shapes_copy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.h_hape is not None
                    and self.h_shape_is_selected
                    and not self.moving_shape
                ):
                    self.selection_changed.emit(
                        [x for x in self.selected_shapes if x != self.h_hape]
                    )

        self.store_moving_shape()

    def end_move(self, copy):
        """End of move"""
        assert self.selected_shapes and self.selected_shapes_copy
        assert len(self.selected_shapes_copy) == len(self.selected_shapes)
        if copy:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.shapes.append(shape)
                self.selected_shapes[i].selected = False
                self.selected_shapes[i] = shape
        else:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.selected_shapes[i].points = shape.points
        self.selected_shapes_copy = []
        self.repaint()
        self.store_shapes()
        return True

    def hide_background_shapes(self, value):
        """Set hide background - hide other shapes when some shapes are selected"""
        self.hide_backround = value
        if self.selected_shapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.set_hiding(True)
            self.update()

    def set_hiding(self, enable=True):
        """Set background hiding"""
        self._hide_backround = self.hide_backround if enable else False

    def can_close_shape(self):
        """Check if a shape can be closed (number of points > 2)"""
        return self.drawing() and self.current and len(self.current) > 2

    # QT Overload
    def mouseDoubleClickEvent(self, _):
        """Mouse double click event"""
        if self.is_loading:
            return
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.can_close_shape()
            and len(self.current) > 3
        ):
            self.current.pop_point()
            self.finalise()

    def select_shapes(self, shapes):
        """Select some shapes"""
        self.set_hiding()
        self.selection_changed.emit(shapes)
        self.update()

    def select_shape_point(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selected_vertex():  # A vertex is marked for selection.
            index, shape = self.h_vertex, self.h_hape
            shape.highlight_vertex(index, shape.MOVE_VERTEX)
            if shape.shape_type == "rotation":
                self.set_hiding()
                if shape not in self.selected_shapes:
                    if multiple_selection_mode:
                        self.selection_changed.emit(
                            self.selected_shapes + [shape]
                        )
                    else:
                        self.selection_changed.emit([shape])
                    self.h_shape_is_selected = False
                else:
                    self.h_shape_is_selected = True
                self.calculate_offsets(point)
                return

        else:
            for shape in reversed(self.shapes):
                if not self.is_visible(shape):
                    continue
                if shape.is_mask():
                    if shape.contains_point(point):
                        # 이하 기존 선택 로직 복사
                        self.set_hiding()
                        if shape not in self.selected_shapes:
                            if multiple_selection_mode:
                                self.selection_changed.emit(self.selected_shapes + [shape])
                            else:
                                self.selection_changed.emit([shape])
                            self.h_shape_is_selected = False
                        else:
                            self.h_shape_is_selected = True
                        self.calculate_offsets(point)
                        return
                elif len(shape.points) > 1 and shape.contains_point(point):
                    # 이하 기존 선택 로직
                    self.set_hiding()
                    if shape not in self.selected_shapes:
                        if multiple_selection_mode:
                            self.selection_changed.emit(self.selected_shapes + [shape])
                        else:
                            self.selection_changed.emit([shape])
                        self.h_shape_is_selected = False
                    else:
                        self.h_shape_is_selected = True
                    self.calculate_offsets(point)
                    return
        self.deselect_shape()

    def calculate_offsets(self, point):
        """Calculate offsets of a point to pixmap borders"""
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selected_shapes:
            rect = s.bounding_rect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def get_adjoint_points(self, theta, p3, p1, index):
        a1 = math.tan(theta)
        if a1 == 0:
            if index % 2 == 0:
                p2 = QtCore.QPointF(p3.x(), p1.y())
                p4 = QtCore.QPointF(p1.x(), p3.y())
            else:
                p4 = QtCore.QPointF(p3.x(), p1.y())
                p2 = QtCore.QPointF(p1.x(), p3.y())
        else:
            a3 = a1
            a2 = -1 / a1
            a4 = -1 / a1
            b1 = p1.y() - a1 * p1.x()
            b2 = p1.y() - a2 * p1.x()
            b3 = p3.y() - a1 * p3.x()
            b4 = p3.y() - a2 * p3.x()

            if index % 2 == 0:
                p2 = self.get_cross_point(a1, b1, a4, b4)
                p4 = self.get_cross_point(a2, b2, a3, b3)
            else:
                p4 = self.get_cross_point(a1, b1, a4, b4)
                p2 = self.get_cross_point(a2, b2, a3, b3)

        return p2, p3, p4

    @staticmethod
    def get_cross_point(a1, b1, a2, b2):
        x = (b2 - b1) / (a1 - a2)
        y = (a1 * b2 - a2 * b1) / (a1 - a2)
        return QtCore.QPointF(x, y)

    def bounded_move_vertex(self, pos):
        """Move a vertex. Adjust position to be bounded by pixmap border"""
        index, shape = self.h_vertex, self.h_hape
        point = shape[index]
        if (
            self.out_off_pixmap(pos)
            and shape.shape_type not in self.allowed_oop_shape_types
        ):
            pos = self.intersection_point(point, pos)

        if shape.shape_type == "rotation":
            sindex = (index + 2) % 4
            # Get the other 3 points after transformed
            p2, p3, p4 = self.get_adjoint_points(
                shape.direction, shape[sindex], pos, index
            )
            # if (
            #     self.out_off_pixmap(p2)
            #     or self.out_off_pixmap(p3)
            #     or self.out_off_pixmap(p4)
            # ):
            #     # No need to move if one pixal out of map
            #     return
            # Move 4 pixal one by one
            shape.move_vertex_by(index, pos - point)
            lindex = (index + 1) % 4
            rindex = (index + 3) % 4
            shape[lindex] = p2
            shape[rindex] = p4
            shape.close()
        elif shape.shape_type == "rectangle":
            shift_pos = pos - point
            shape.move_vertex_by(index, shift_pos)
            left_index = (index + 1) % 4
            right_index = (index + 3) % 4
            left_shift = None
            right_shift = None
            if index % 2 == 0:
                right_shift = QtCore.QPointF(shift_pos.x(), 0)
                left_shift = QtCore.QPointF(0, shift_pos.y())
            else:
                left_shift = QtCore.QPointF(shift_pos.x(), 0)
                right_shift = QtCore.QPointF(0, shift_pos.y())
            shape.move_vertex_by(right_index, right_shift)
            shape.move_vertex_by(left_index, left_shift)
        else:
            shape.move_vertex_by(index, pos - point)

    def bounded_move_shapes(self, shapes, pos):
        """Move shapes. Adjust position to be bounded by pixmap border"""
        shape_types = []
        for shape in shapes:
            if shape.shape_type in self.allowed_oop_shape_types:
                shape_types.append(shape.shape_type)

        if self.out_off_pixmap(pos) and len(shape_types) == 0:
            return False  # No need to move
        if len(shape_types) > 0 and len(shapes) != len(shape_types):
            return False

        if len(shape_types) == 0:
            o1 = pos + self.offsets[0]
            if self.out_off_pixmap(o1):
                pos -= QtCore.QPoint(min(0, int(o1.x())), min(0, int(o1.y())))
            o2 = pos + self.offsets[1]
            if self.out_off_pixmap(o2):
                pos += QtCore.QPoint(
                    min(0, int(self.pixmap.width() - o2.x())),
                    min(0, int(self.pixmap.height() - o2.y())),
                )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prev_point
        if dp:
            for shape in shapes:
                shape.move_by(dp)
            self.prev_point = pos
            return True
        return False

    def rotate_point(self, p, center, theta):
        order = p - center
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = -sinTheta * order.x() + cosTheta * order.y()
        pRes = QtCore.QPointF(center.x() + pResx, center.y() + pResy)
        return pRes

    def bounded_rotate_shapes(self, i, shape, theta):
        """Rotate shapes. Adjust position to be bounded by pixmap border"""
        new_shape = deepcopy(shape)
        if len(shape.points) == 2:
            new_shape.points[0] = shape.points[0]
            new_shape.points[1] = QtCore.QPointF(
                (shape.points[0].x() + shape.points[1].x()) / 2,
                shape.points[0].y(),
            )
            new_shape.points.append(shape.points[1])
            new_shape.points.append(
                QtCore.QPointF(
                    shape.points[1].x(),
                    (shape.points[0].y() + shape.points[1].y()) / 2,
                )
            )
        center = QtCore.QPointF(
            (new_shape.points[0].x() + new_shape.points[2].x()) / 2,
            (new_shape.points[0].y() + new_shape.points[2].y()) / 2,
        )
        for j, p in enumerate(new_shape.points):
            pos = self.rotate_point(p, center, theta)
            # TODO: Reserved for now
            # if self.out_off_pixmap(pos):
            #     return False  # No need to rotate
            new_shape.points[j] = pos
        new_shape.direction = (new_shape.direction - theta) % (2 * math.pi)
        self.selected_shapes[i].points = new_shape.points
        self.selected_shapes[i].direction = new_shape.direction
        return True

    def deselect_shape(self):
        """Deselect all shapes"""
        if self.selected_shapes:
            self.set_hiding(False)
            self.selection_changed.emit([])
            self.h_shape_is_selected = False
            self.update()

    def delete_selected(self):
        """Remove selected shapes"""
        deleted_shapes = []
        if self.selected_shapes:
            for shape in self.selected_shapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.store_shapes()
            self.selected_shapes = []
            self.update()
        return deleted_shapes

    def delete_shape(self, shape):
        """Remove a specific shape"""
        if shape in self.selected_shapes:
            self.selected_shapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.store_shapes()
        self.update()

    def duplicate_selected_shapes(self):
        """Duplicate selected shapes"""
        if self.selected_shapes:
            self.selected_shapes_copy = [
                s.copy() for s in self.selected_shapes
            ]
            self.bounded_shift_shapes(self.selected_shapes_copy)
            self.end_move(copy=True)
        return self.selected_shapes

    def bounded_shift_shapes(self, shapes):
        """
        Shift shapes by an offset. Adjust positions to be bounded
        by pixmap borders
        """
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.prev_point = point
        if not self.bounded_move_shapes(shapes, point - offset):
            self.bounded_move_shapes(shapes, point + offset)

    # QT Overload
    def paintEvent(self, event):  # noqa: C901
        """Paint event for canvas"""
        if (
            self.pixmap is None
            or self.pixmap.width() == 0
            or self.pixmap.height() == 0
        ):
            super().paintEvent(event)
            return

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)

        p.scale(self.scale, self.scale)
        p.translate(self.offset_to_center())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale

        for shape in self.shapes:
            if getattr(shape, "mask", None) is not None:
                # Pass the _get_rgb_by_label function if available
                get_rgb_func = getattr(self.parent, "_get_rgb_by_label", None) if hasattr(self, 'parent') else None
                shape.paint_mask(p, get_rgb_func)
                continue
            shape.paint(p)

        # Draw loading/waiting screen
        if self.is_loading:
            # Draw a semi-transparent rectangle
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 20))
            p.drawRect(self.pixmap.rect())

            # Draw a spinning wheel
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setBrush(Qt.NoBrush)
            p.save()
            p.translate(self.pixmap.width() / 2, self.pixmap.height() / 2 - 50)
            p.rotate(self.loading_angle)
            p.drawEllipse(-20, -20, 40, 40)
            p.drawLine(0, 0, 0, -20)
            p.restore()
            self.loading_angle += 30
            if self.loading_angle >= 360:
                self.loading_angle = 0

            # Draw the loading text
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setFont(QtGui.QFont("Arial", 20))
            p.drawText(
                self.pixmap.rect(),
                Qt.AlignCenter,
                self.loading_text,
            )
            p.end()
            self.update()
            return

        # Draw groups
        if self.show_groups:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.SolidLine)
            p.setPen(pen)
            grouped_shapes = {}
            for shape in self.shapes:
                if not shape.visible:
                    continue
                if shape.group_id is None:
                    continue
                if shape.group_id not in grouped_shapes:
                    grouped_shapes[shape.group_id] = []
                grouped_shapes[shape.group_id].append(shape)

            for group_id in grouped_shapes:
                shapes = grouped_shapes[group_id]
                min_x = float("inf")
                min_y = float("inf")
                max_x = 0
                max_y = 0
                for shape in shapes:
                    rect = shape.bounding_rect()
                    if shape.shape_type == "point":
                        points = shape.points[0]
                        min_x = min(min_x, points.x())
                        min_y = min(min_y, points.y())
                        max_x = max(max_x, points.x())
                        max_y = max(max_y, points.y())
                    else:
                        min_x = min(min_x, rect.x())
                        min_y = min(min_y, rect.y())
                        max_x = max(max_x, rect.x() + rect.width())
                        max_y = max(max_y, rect.y() + rect.height())
                    group_color = LABEL_COLORMAP[
                        int(group_id) % len(LABEL_COLORMAP)
                    ]
                    pen.setStyle(Qt.SolidLine)
                    pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                    pen.setColor(QtGui.QColor(*group_color))
                    p.setPen(pen)

                    # Calculate the center point of the bounding rectangle
                    cx = rect.x() + rect.width() / 2
                    cy = rect.y() + rect.height() / 2
                    triangle_radius = max(1, int(round(3.0 / Shape.scale)))

                    # Define the points of the triangle
                    triangle_points = [
                        QtCore.QPointF(cx, cy - triangle_radius),
                        QtCore.QPointF(
                            cx - triangle_radius, cy + triangle_radius
                        ),
                        QtCore.QPointF(
                            cx + triangle_radius, cy + triangle_radius
                        ),
                    ]

                    # Draw the triangle
                    p.drawPolygon(triangle_points)

                pen.setStyle(Qt.DashLine)
                pen.setWidth(max(1, int(round(1.0 / Shape.scale))))
                pen.setColor(QtGui.QColor("#EEEEEE"))
                p.setPen(pen)
                wrap_rect = QtCore.QRectF(
                    min_x, min_y, max_x - min_x, max_y - min_y
                )
                p.drawRect(wrap_rect)

        # Draw KIE linking
        if self.show_linking:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.SolidLine)
            p.setPen(pen)
            gid2point = {}
            linking_pairs = []
            group_color = (255, 128, 0)
            for shape in self.shapes:
                if not shape.visible:
                    continue

                try:
                    linking_pairs += shape.kie_linking
                except Exception:
                    pass

                if shape.group_id is None or shape.shape_type not in [
                    "rectangle",
                    "polygon",
                    "rotation",
                ]:
                    continue
                rect = shape.bounding_rect()
                cx = rect.x() + (rect.width() / 2.0)
                cy = rect.y() + (rect.height() / 2.0)
                gid2point[shape.group_id] = (cx, cy)

            for linking in linking_pairs:
                pen.setStyle(Qt.SolidLine)
                pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                pen.setColor(QtGui.QColor(*group_color))
                p.setPen(pen)
                key, value = linking
                # Adapt to the 'ungroup_selected_shapes' operation
                if key not in gid2point or value not in gid2point:
                    continue
                kp, vp = gid2point[key], gid2point[value]
                # Draw a link from key point to value point
                p.drawLine(QtCore.QPointF(*kp), QtCore.QPointF(*vp))
                # Draw the triangle arrowhead
                arrow_size = max(
                    1, int(round(10.0 / Shape.scale))
                )  # Size of the arrowhead
                angle = math.atan2(
                    vp[1] - kp[1], vp[0] - kp[0]
                )  # Angle towards the value point
                arrow_points = [
                    QtCore.QPointF(vp[0], vp[1]),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle - math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle - math.pi / 6),
                    ),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle + math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle + math.pi / 6),
                    ),
                ]
                p.drawPolygon(arrow_points)

        # Draw degrees
        for shape in self.shapes:
            if (
                shape.selected or not self._hide_backround
            ) and self.is_visible(shape):
                # Skip mask shapes as they are already drawn above
                if shape.is_mask() and shape.mask is not None:
                    continue
                shape.fill = self._fill_drawing and (
                    shape.selected or shape == self.h_hape
                )
                shape.paint(p)


            if (
                shape.shape_type == "rotation"
                and len(shape.points) == 4
                and self.is_visible(shape)
            ):
                d = shape.point_size / shape.scale
                center = QtCore.QPointF(
                    (shape.points[0].x() + shape.points[2].x()) / 2,
                    (shape.points[0].y() + shape.points[2].y()) / 2,
                )
                if self.show_degrees:
                    degrees = str(int(math.degrees(shape.direction))) + "°"
                    p.setFont(
                        QtGui.QFont(
                            "Arial",
                            int(max(6.0, int(round(8.0 / Shape.scale)))),
                        )
                    )
                    pen = QtGui.QPen(
                        QtGui.QColor("#FF9900"), 8, QtCore.Qt.SolidLine
                    )
                    p.setPen(pen)
                    fm = QtGui.QFontMetrics(p.font())
                    rect = fm.boundingRect(degrees)
                    p.fillRect(
                        int(rect.x() + center.x() - d),
                        int(rect.y() + center.y() + d),
                        int(rect.width()),
                        int(rect.height()),
                        QtGui.QColor("#FF9900"),
                    )
                    pen = QtGui.QPen(
                        QtGui.QColor("#FFFFFF"), 7, QtCore.Qt.SolidLine
                    )
                    p.setPen(pen)
                    p.drawText(
                        int(center.x() - d),
                        int(center.y() + d),
                        degrees,
                    )
                else:
                    cp = QtGui.QPainterPath()
                    cp.addRect(
                        int(center.x() - d / 2),
                        int(center.y() - d / 2),
                        int(d),
                        int(d),
                    )
                    p.drawPath(cp)
                    p.fillPath(cp, QtGui.QColor(255, 153, 0, 255))

        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selected_shapes_copy:
            for s in self.selected_shapes_copy:
                s.paint(p)

        if (
            self.fill_drawing()
            and self.create_mode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.add_point(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        # Draw texts
        if self.show_texts:
            text_color = "#FFFFFF"
            background_color = "#007BFF"
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            pen = QtGui.QPen(QtGui.QColor(background_color), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                description = shape.description
                if description:
                    bbox = shape.bounding_rect()
                    fm = QtGui.QFontMetrics(p.font())
                    rect = fm.boundingRect(description)
                    p.fillRect(
                        int(rect.x() + bbox.x()),
                        int(rect.y() + bbox.y()),
                        int(rect.width()),
                        int(rect.height()),
                        QtGui.QColor(background_color),
                    )
                    p.drawText(
                        int(bbox.x()),
                        int(bbox.y()),
                        description,
                    )
            pen = QtGui.QPen(QtGui.QColor(text_color), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                description = shape.description
                if description:
                    bbox = shape.bounding_rect()
                    p.drawText(
                        int(bbox.x()),
                        int(bbox.y()),
                        description,
                    )

        # Draw labels
        if self.show_labels:
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            labels = []
            for shape in self.shapes:
                if not shape.visible:
                    continue
                d_react = shape.point_size / shape.scale
                d_text = 1.5
                if not shape.visible:
                    continue
                if shape.label in [
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                ]:
                    continue
                label_text = (
                    (
                        f"id:{shape.group_id} "
                        if shape.group_id is not None
                        else ""
                    )
                    + (f"{shape.label}")
                    + (
                        f" {float(shape.score):.2f}"
                        if (shape.score is not None and self.show_scores)
                        else ""
                    )
                )
                if not label_text:
                    continue
                fm = QtGui.QFontMetrics(p.font())
                bound_rect = fm.boundingRect(label_text)
                if shape.shape_type in ["rectangle", "polygon", "rotation", "mask"]:
                    try:
                        bbox = shape.bounding_rect()
                    except IndexError:
                        continue
                    rect = QtCore.QRect(
                        int(bbox.x()),
                        int(bbox.y()),
                        int(bound_rect.width()),
                        int(bound_rect.height()),
                    )
                    text_pos = QtCore.QPoint(
                        int(bbox.x()),
                        int(bbox.y() + bound_rect.height() - d_text),
                    )
                elif shape.shape_type in [
                    "circle",
                    "line",
                    "linestrip",
                    "point",
                ]:
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]
                    rect = QtCore.QRect(
                        int(point.x() + d_react),
                        int(point.y() - 15),
                        int(bound_rect.width()),
                        int(bound_rect.height()),
                    )
                    text_pos = QtCore.QPoint(
                        int(point.x()),
                        int(point.y() - 15 + bound_rect.height() - d_text),
                    )
                else:
                    continue
                labels.append((shape, rect, text_pos, label_text))

            pen = QtGui.QPen(QtGui.QColor("#FFA500"), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape, rect, _, _ in labels:
                if not shape.visible:
                    continue
                p.fillRect(rect, shape.line_color)

            pen = QtGui.QPen(QtGui.QColor("#000000"), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape, _, text_pos, label_text in labels:
                if not shape.visible:
                    continue
                p.drawText(text_pos, label_text)

        # Draw mouse coordinates
        if self.cross_line_show:
            pen = QtGui.QPen(
                QtGui.QColor(self.cross_line_color),
                max(1, int(round(self.cross_line_width / Shape.scale))),
                Qt.DashLine,
            )
            p.setPen(pen)
            p.setOpacity(self.cross_line_opacity)
            p.drawLine(
                QtCore.QPointF(self.prev_move_point.x(), 0),
                QtCore.QPointF(self.prev_move_point.x(), self.pixmap.height()),
            )
            p.drawLine(
                QtCore.QPointF(0, self.prev_move_point.y()),
                QtCore.QPointF(self.pixmap.width(), self.prev_move_point.y()),
            )

        # 브러시 프리뷰 원 (화면에서 일정한 크기 유지)
        if self.is_brush_mode:
            painter = self._painter  
            # 스케일 변환 보정으로 화면에서 일정한 크기 유지
            r = self.brush_radius / self.scale
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.drawEllipse(self.prev_move_point, r, r)

        # print("is_brush_mode:", self.is_brush_mode, "prev_move_point:", self.prev_move_point)
        p.end()

    def edit_mask_with_brush(self, shape, pos, radius, add=True):
        """브러시 드래그 시 호출. pos는 widget 좌표, transform_pos로 이미지 좌표로 변환하세요."""

        # 1) polygon/rect/rotation 일 때도 한번만 mask로 변환
        if shape.shape_type != "mask":
            # 원래 타입 저장 (이미 저장되어 있으면 유지)
            if not hasattr(shape, '_original_shape_type'):
                shape._original_shape_type = shape.shape_type
            
            h, w = self.pixmap.height(), self.pixmap.width()
            shape.mask = polygon_to_mask(
                [(int(p.x()), int(p.y())) for p in shape.points],
                (h, w),
            )
            shape.shape_type = "mask"
            shape.points = []

        # 2) 성능 최적화: 직접 마스크 수정 (복사 없이)
        old_sum = shape.mask.sum()  # 변경 감지용
        
        # 3) 실제 브러시 연산 (기존 마스크에 직접 적용) - 전달받은 radius 사용
        shape.mask = apply_brush_to_mask(shape.mask, pos.x(), pos.y(), radius=radius, add=add)
        
        # 4) 마스크가 완전히 사라지면 도형 삭제
        new_sum = shape.mask.sum()
        if new_sum == 0:
            # canvas에서 shape 제거
            self.delete_shape(shape)
            # label_list에서도 제거
            if hasattr(self.parent, 'remove_labels'):
                self.parent.remove_labels([shape])
            # 브러시 타겟 shape 정리
            if self._brush_target_shape == shape:
                self._brush_target_shape = None
            self.parent.status("브러시로 도형이 완전히 삭제되었습니다.")
            return

        # 5) 변경사항이 있으면 반영 (픽셀 합계 비교로 빠른 체크)
        if old_sum != new_sum:
            # 캐시 무효화 (렌더링 성능을 위해 유지)
            self._mask_qimage_cache.pop(shape, None)
            self._mask_overlay_cache.pop(shape, None)
            # dirty flag
            self.brush_modified = True
            
            # 선택 상태 명시적으로 유지 (한 번만)
            if shape not in self.selected_shapes:
                self.selected_shapes.append(shape)

            self.update()

    def _delayed_brush_update(self):
        """브러시 업데이트 최적화: 지연된 업데이트"""
        # 항상 업데이트 (라벨 표시 등을 위해)
        self.update()

    def _batch_process_brush_stroke(self, stroke_points):
        """브러시 스트로크 배치 처리로 성능 최적화"""
        if not stroke_points or not self._brush_target_shape:
            return
            
        # 한 번에 여러 포인트 처리
        shape = self._brush_target_shape
        for pos, add in stroke_points:
            # 직접 마스크 수정 (중간 체크 없이)
            shape.mask = apply_brush_to_mask(
                shape.mask, pos.x(), pos.y(), 
                radius=max(1, int(self.brush_radius / self.scale)), add=add
            )
        
        # 배치 처리 후 한 번만 캐시 무효화
        self._mask_qimage_cache.pop(shape, None)
        self._mask_overlay_cache.pop(shape, None)
        self.brush_modified = True

    def transform_pos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones.""" 
        return point / self.scale - self.offset_to_center()

    def offset_to_center(self):
        """Calculate offset to the center"""
        if self.pixmap is None:
            return QtCore.QPointF()
        s = self.scale
        area = super().size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        area_width, area_height = area.width(), area.height()
        x = (area_width - w) / (2 * s) if area_width > w else 0
        y = (area_height - h) / (2 * s) if area_height > h else 0
        return QtCore.QPointF(x, y)

    def out_off_pixmap(self, p):
        """Check if a position is out of pixmap"""
        if self.pixmap is None:
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        """Finish drawing for a shape"""
        assert self.current
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            self.current.label = self.auto_labeling_mode.edit_mode
        # TODO(vietanhdev): Temporrally fix. Need to refactor
        if self.current.label is None:
            self.current.label = ""
        self.current.close()
        self.shapes.append(self.current)
        self.store_shapes()
        self.current = None
        self.set_hiding(False)
        self.new_shape.emit()
        self.update()
        if self.is_auto_labeling:
            self.update_auto_labeling_marks()

    def update_auto_labeling_marks(self):
        """Update the auto labeling marks"""
        marks = []
        for shape in self.shapes:
            if shape.label == AutoLabelingMode.ADD:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 1,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 1,
                        }
                    )
            elif shape.label == AutoLabelingMode.REMOVE:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 0,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 0,
                        }
                    )

        self.auto_labeling_marks_updated.emit(marks)

    def close_enough(self, p1, p2):
        """Check if 2 points are close enough (by an threshold epsilon)"""
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersection_point(self, p1, p2):
        """Cycle through each image edge in clockwise fashion,
        and find the one intersecting the current line segment.
        """
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        _, i, (x, y) = min(self.intersecting_edges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        x3, y3 = int(x3), int(y3)
        x4, y4 = int(x4), int(y4)
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            # y3 == y4
            return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(int(x), int(y))

    def intersecting_edges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    # QT Overload
    def sizeHint(self):
        """Get size hint"""
        return self.minimumSizeHint()

    # QT Overload
    def minimumSizeHint(self):
        """Get minimum size hint"""
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super().minimumSizeHint()

    # QT Overload
    def wheelEvent(self, ev: QWheelEvent):
        """Mouse wheel event"""
        mods = ev.modifiers()
        delta = ev.angleDelta()

        if (
            self.editing()
            and self.enable_wheel_rectangle_editing
            and len(self.selected_shapes) == 1
            and self.selected_shapes[0].shape_type == "rectangle"
            and not (QtCore.Qt.ControlModifier & int(mods))
        ):

            try:
                pos = self.transform_pos(ev.posF())
            except AttributeError:
                pos = self.transform_pos(ev.localPos())

            shape = self.selected_shapes[0]
            wheel_up = delta.y() > 0

            if shape.contains_point(pos):
                self._scale_rectangle(shape, wheel_up)
            else:
                self._adjust_rectangle_edge(shape, pos, wheel_up)

            self.store_shapes()
            self.shape_moved.emit()
            self.update()
            ev.accept()
            return

        if QtCore.Qt.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoom_request.emit(delta.y(), ev.pos())
        else:
            # scroll
            self.scroll_request.emit(delta.x(), QtCore.Qt.Horizontal, 0)
            self.scroll_request.emit(delta.y(), QtCore.Qt.Vertical, 0)
        ev.accept()

    def _scale_rectangle(self, shape, scale_up):
        """Scale rectangle from center while keeping within image boundaries"""
        if len(shape.points) < 4:
            return

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        x_coords = [p.x() for p in shape.points]
        y_coords = [p.y() for p in shape.points]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        center = QtCore.QPointF(center_x, center_y)

        scale_factor = (
            1.0 + self.rect_scale_step
            if scale_up
            else 1.0 - self.rect_scale_step
        )
        scale_factor = max(0.1, scale_factor)

        new_points = []
        for i in range(len(shape.points)):
            point = shape.points[i]
            offset = point - center
            scaled_offset = offset * scale_factor
            new_point = center + scaled_offset

            if (
                new_point.x() < 0
                or new_point.x() >= img_width
                or new_point.y() < 0
                or new_point.y() >= img_height
            ):
                return

            new_points.append(new_point)

        for i, new_point in enumerate(new_points):
            shape.points[i] = new_point

    def _adjust_rectangle_edge(self, shape, cursor_pos, move_outward):
        """Adjust the rectangle edge closest to cursor position within image boundaries"""
        if len(shape.points) < 4:
            return

        rect = shape.bounding_rect()
        min_x, max_x = rect.left(), rect.right()
        min_y, max_y = rect.top(), rect.bottom()

        distances = {}

        if cursor_pos.x() < min_x:
            distances["left"] = min_x - cursor_pos.x()
        elif cursor_pos.x() > max_x:
            distances["right"] = cursor_pos.x() - max_x
        else:
            distances["left"] = abs(cursor_pos.x() - min_x)
            distances["right"] = abs(cursor_pos.x() - max_x)

        if cursor_pos.y() < min_y:
            distances["top"] = min_y - cursor_pos.y()
        elif cursor_pos.y() > max_y:
            distances["bottom"] = cursor_pos.y() - max_y
        else:
            distances["top"] = abs(cursor_pos.y() - min_y)
            distances["bottom"] = abs(cursor_pos.y() - max_y)

        if (
            cursor_pos.x() < min_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            closest_edge = "left"
        elif (
            cursor_pos.x() > max_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            closest_edge = "right"
        elif (
            cursor_pos.y() < min_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            closest_edge = "top"
        elif (
            cursor_pos.y() > max_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            closest_edge = "bottom"
        else:
            closest_edge = min(distances, key=distances.get)

        step = (
            self.rect_adjust_step if move_outward else -self.rect_adjust_step
        )

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        for i, point in enumerate(shape.points):
            new_point = None

            if closest_edge == "left" and abs(point.x() - min_x) < 1e-6:
                new_x = max(0, point.x() - step)
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "right" and abs(point.x() - max_x) < 1e-6:
                new_x = min(img_width - 1, point.x() + step)
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "top" and abs(point.y() - min_y) < 1e-6:
                new_y = max(0, point.y() - step)
                new_point = QtCore.QPointF(point.x(), new_y)
            elif closest_edge == "bottom" and abs(point.y() - max_y) < 1e-6:
                new_y = min(img_height - 1, point.y() + step)
                new_point = QtCore.QPointF(point.x(), new_y)

            if new_point is not None:
                shape.points[i] = new_point

    def move_by_keyboard(self, offset):
        """Move selected shapes by an offset (using keyboard)"""
        if self.selected_shapes:
            self.bounded_move_shapes(
                self.selected_shapes, self.prev_point + offset
            )
            self.repaint()
            self.moving_shape = True

    def rotate_by_keyboard(self, theta):
        """Rotate selected shapes by an theta (using keyboard)"""
        if self.selected_shapes:
            for i, shape in enumerate(self.selected_shapes):
                if shape._shape_type == "rotation":
                    self.bounded_rotate_shapes(i, shape, theta)
                    self.repaint()
                    self.rotating_shape = True

    # QT Overload
    def keyPressEvent(self, ev):
        """Key press event"""
        modifiers = ev.modifiers()
        key = ev.key()

        
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawing_polygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.can_close_shape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.move_by_keyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.move_by_keyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.move_by_keyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.move_by_keyboard(QtCore.QPointF(MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Z:
                self.rotate_by_keyboard(LARGE_ROTATION_INCREMENT)
            elif key == QtCore.Qt.Key_X:
                self.rotate_by_keyboard(SMALL_ROTATION_INCREMENT)
            elif key == QtCore.Qt.Key_C:
                self.rotate_by_keyboard(-SMALL_ROTATION_INCREMENT)
            elif key == QtCore.Qt.Key_V:
                self.rotate_by_keyboard(-LARGE_ROTATION_INCREMENT)
            # brush mode toggle
            elif key == QtCore.Qt.Key_M:
                self.set_brush_mode(not self.is_brush_mode)
                return

    # QT Overload
    def keyReleaseEvent(self, ev):
        """Key release event"""
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            # NOTE: Temporary fix to avoid ValueError
            # when the selected shape is not in the shapes list
            if (
                (self.moving_shape or self.rotating_shape)
                and self.selected_shapes
                and self.selected_shapes[0] in self.shapes
            ):
                index = self.shapes.index(self.selected_shapes[0])
                if (
                    self.shapes_backups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.store_shapes()
                    if self.moving_shape:
                        self.shape_moved.emit()
                    if self.rotating_shape:
                        self.shape_rotated.emit()

                if self.moving_shape:
                    self.moving_shape = False
                if self.rotating_shape:
                    self.rotating_shape = False

    def set_last_label(self, text, flags):
        """Set label and flags for last shape"""
        assert text
        if self.is_auto_labeling:
            self.shapes[-1].label = self.auto_labeling_mode.edit_mode
        else:
            self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapes_backups.pop()
        self.store_shapes()
        return self.shapes[-1]

    def undo_last_line(self):
        """Undo last line"""
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.set_open()
        if self.create_mode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.create_mode in ["rectangle", "line", "circle", "rotation"]:
            self.current.points = self.current.points[0:1]
        elif self.create_mode == "point":
            self.current = None
        self.drawing_polygon.emit(True)

    def undo_last_point(self):
        """Undo last point"""
        if not self.current or self.current.is_closed():
            return
        self.current.pop_point()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawing_polygon.emit(False)
        self.update()

    def load_pixmap(self, pixmap, clear_shapes=True):
        """Load pixmap"""
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
        # 이미지 크기에 따른 브러시 크기 자동 설정 (매번 초기화)
        # 최적의 브러시 크기 계산
        brush_radius, slider_value = self.calculate_optimal_brush_size(pixmap.width(), pixmap.height())
        self.brush_radius = brush_radius
        
        # 슬라이더 값도 자동 설정 (무한 루프 방지)
        if hasattr(self, 'parent') and hasattr(self.parent, 'brush_options_panel'):
            slider = self.parent.brush_options_panel.slider
            slider.blockSignals(True)
            slider.setValue(slider_value)
            slider.blockSignals(False)
        
        self.update()

    def load_shapes(self, shapes, replace=True):
        """Load shapes"""
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.store_shapes()
        self.current = None
        self.h_hape = None
        self.h_vertex = None
        self.h_edge = None
        self.update()

    def set_shape_visible(self, shape, value):
        """Set visibility for a shape"""
        self.visible[shape] = value
        self.update()

    def current_cursor(self):
        """Current cursor"""
        cursor = QtWidgets.QApplication.overrideCursor()
        cursor = cursor.shape() if cursor else None

        return cursor

    def override_cursor(self, cursor):
        """Override cursor"""
        current_cursor = self.current_cursor()
        if current_cursor != cursor:
            self._cursor = cursor
            if current_cursor is None:
                QtWidgets.QApplication.setOverrideCursor(cursor)
            else:
                QtWidgets.QApplication.changeOverrideCursor(cursor)

    def restore_cursor(self):
        """Restore override cursor"""
        QtWidgets.QApplication.restoreOverrideCursor()

    def reset_state(self):
        """Clear shapes and pixmap"""
        self.restore_cursor()
        self.pixmap = None
        self.shapes_backups = []
        self.is_move_editing = False
        self.update()

    def set_cross_line(self, show, width, color, opacity):
        """Set cross line options"""
        self.cross_line_show = show
        self.cross_line_width = width
        self.cross_line_color = color
        self.cross_line_opacity = opacity
        self.update()

    def gen_new_group_id(self):
        """Generate new shape's group_id based on current shapes"""
        max_group_id = 0
        for shape in self.shapes:
            if shape.group_id is not None:
                max_group_id = max(max_group_id, shape.group_id)
        return max_group_id + 1

    def merge_group_ids(self, group_ids, new_group_id):
        """Merge multiple shapes' group_id into a new one"""
        for shape in self.shapes:
            if shape.group_id in group_ids:
                shape.group_id = new_group_id

    def group_selected_shapes(self):
        """Group selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        has_non_group_shape = False
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)
            else:
                has_non_group_shape = True

        # If there is at least 1 shape having a group id,
        # use that id as the new group id. Otherwise, generate a new group_id
        new_group_id = None
        if len(group_ids) > 0:
            new_group_id = min(group_ids)
        else:
            new_group_id = self.gen_new_group_id()

        # Merge group ids
        if len(group_ids) > 1:
            self.merge_group_ids(
                group_ids=group_ids, new_group_id=new_group_id
            )
        # Assign new_group_id to non-group shapes
        if has_non_group_shape:
            for shape in self.selected_shapes:
                if shape.group_id is None:
                    shape.group_id = new_group_id

        self.update()

    def ungroup_selected_shapes(self):
        """Ungroup selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)

        for group_id in group_ids:
            for shape in self.shapes:
                if shape.group_id == group_id:
                    shape.group_id = None

        self.update()

    def set_brush_mode(self, enabled: bool, radius: int = 10):        
        self.is_brush_mode = enabled
        
        # 브러시 모드를 켤 때는 현재 슬라이더 값을 사용
        if enabled:
            self.set_editing(True)
            self.override_cursor(QtCore.Qt.BlankCursor)  # 마우스 커서 숨김
            if hasattr(self, 'parent') and hasattr(self.parent, 'brush_options_panel'):
                slider = self.parent.brush_options_panel.slider
                self.brush_radius = slider.value() / 10.0
            else:
                self.brush_radius = radius / 10.0
        else:
            self.override_cursor(CURSOR_DEFAULT)  

        # 브러시 모드 OFF일 때는 _brush_target_shape를 우선 사용
        if not enabled and self._brush_target_shape is not None:
            target_shape = self._brush_target_shape
        else:
            target_shape = self.selected_shapes[0] if self.selected_shapes else None
        if enabled:
            # polygon/rectangle/rotation → mask 변환
            if target_shape is not None and target_shape.shape_type in ["polygon", "rectangle", "rotation"]:
                
                h, w = self.pixmap.height(), self.pixmap.width()
                mask = polygon_to_mask([(p.x(), p.y()) for p in target_shape.points], (h, w))
                from ..shape import Shape
                mask_shape = Shape(
                    shape_type="mask",
                    label=target_shape.label,
                    group_id=target_shape.group_id,
                    mask=mask
                )
                # 원본 shape_type 저장 (이미 polygon인 경우 _original_shape_type 사용)
                if hasattr(target_shape, '_original_shape_type'):
                    mask_shape._original_shape_type = target_shape._original_shape_type
                else:
                    mask_shape._original_shape_type = target_shape.shape_type
                # 색상 정보 복사
                mask_shape.line_color = target_shape.line_color
                mask_shape.fill_color = target_shape.fill_color
                mask_shape.select_line_color = target_shape.select_line_color
                mask_shape.select_fill_color = target_shape.select_fill_color
                mask_shape.vertex_fill_color = target_shape.vertex_fill_color
                mask_shape.hvertex_fill_color = target_shape.hvertex_fill_color
                if target_shape in self.shapes:
                    self.shapes.remove(target_shape)
                self.shapes.append(mask_shape)
                self.selected_shapes = [mask_shape]
                self._brush_target_shape = mask_shape
                self.selection_changed.emit([mask_shape])  # selection_changed 시그널도 mask로 emit
                self._prev_brush_pos = None
        else:            
            # 모든 도형을 순회하면서 mask 타입인 것들을 원래 타입으로 변환
            shapes_to_remove = []
            shapes_to_add = []

            mask_count = 0
            
            for shape in self.shapes[:]:  # 복사본으로 순회하여 삭제 중 안전성 보장
                # mask 타입이거나, polygon이지만 mask 속성을 가진 경우 모두 변환
                if ((shape.shape_type == "mask" and hasattr(shape, "mask") and shape.mask is not None) or 
                    (shape.shape_type in ["polygon", "rectangle", "rotation"] and hasattr(shape, "mask") and shape.mask is not None)):
                    mask_count += 1
                    # 원래 타입 결정
                    if hasattr(shape, '_original_shape_type'):
                        original_type = shape._original_shape_type
                    else:
                        # _original_shape_type이 없으면 기본적으로 polygon으로 변환
                        original_type = "polygon"
                    
                    mask = shape.mask
                    if mask.sum() == 0:
                        # 빈 마스크는 삭제
                        shapes_to_remove.append(shape)
                        continue
                    
                    from ..shape import Shape
                    points = mask_to_polygon(mask, simplify=False)
                    if points and len(points) >= 3:
                        poly_shape = Shape(
                            shape_type=original_type,
                            label=shape.label,
                            group_id=shape.group_id
                        )
                        poly_shape.points = [QtCore.QPointF(x, y) for x, y in points]
                        
                        # polygon을 닫힌 상태로 만들기
                        poly_shape.close()
                        
                        # 색상 정보 복사
                        poly_shape.line_color = shape.line_color
                        poly_shape.fill_color = shape.fill_color
                        poly_shape.select_line_color = shape.select_line_color
                        poly_shape.select_fill_color = shape.select_fill_color
                        poly_shape.vertex_fill_color = shape.vertex_fill_color
                        poly_shape.hvertex_fill_color = shape.hvertex_fill_color
                        
                        # rotation 타입인 경우 direction 정보도 복사
                        if original_type == "rotation" and hasattr(shape, 'direction'):
                            poly_shape.direction = shape.direction
                        
                        shapes_to_remove.append(shape)
                        shapes_to_add.append(poly_shape)
                        
                        # 현재 선택된 도형이 변환되는 경우, 새로운 도형을 선택 상태로 유지
                        if shape in self.selected_shapes:
                            self.selected_shapes.remove(shape)
                            self.selected_shapes.append(poly_shape)
                    else:
                        # 유효하지 않은 polygon은 삭제
                        shapes_to_remove.append(shape)
            
            # 변환 작업 실행
            for shape in shapes_to_remove:
                if shape in self.shapes:
                    self.shapes.remove(shape)
            for shape in shapes_to_add:
                self.shapes.append(shape)

            # 선택 상태 업데이트
            if self.selected_shapes:
                self.selection_changed.emit(self.selected_shapes)
            
            # label_list 동기화를 위해 시그널 발생 (실제 변환이 있을 때만)
            if shapes_to_remove or shapes_to_add:
                # label_list를 직접 업데이트 (store_shapes 호출 방지)
                if self.parent and hasattr(self.parent, 'label_list'):
                    self.parent.label_list.clear()
                    for shape in self.shapes:
                        self.parent.add_label(shape, update_last_label=False)
            
            # 모든 변환 작업이 완료된 후에 파일 저장 시그널 발생
            if shapes_to_remove or shapes_to_add:
                self.shape_moved.emit()
            
            self._brush_target_shape = None
            self._prev_brush_pos = None
        
        self.update()
        self.brush_mode_changed.emit(enabled)  # 시그널 발생
    

    def set_brush_radius(self, value):
        self.brush_radius = value
        # 슬라이더 값도 동기화 (무한 루프 방지)
        if hasattr(self, 'parent') and hasattr(self.parent, 'brush_options_panel'):
            slider = self.parent.brush_options_panel.slider
            # 슬라이더 값이 다를 때만 업데이트
            if abs(slider.value() - value * 10) > 0.1:
                slider.blockSignals(True)  # 시그널 차단
                slider.setValue(int(value * 10))
                slider.blockSignals(False)  # 시그널 복원
        self.update()  # ← 슬라이더로 크기 바뀌면 즉시 프리뷰도 갱신

    def set_eraser_mode(self, enabled):
        self.eraser_mode = enabled  # eraser_mode 플래그 추가
        # 실제 브러시 동작에서 add/erase 분기 처리 필요

    def calculate_optimal_brush_size(self, image_width, image_height):
        """화면 표시 크기 기준으로 브러시 크기를 계산"""
        # 현재 스케일 팩터 가져오기 (없으면 1.0으로 기본값)
        current_scale = getattr(self, 'scale', 1.0)
        
        # 화면에서 실제로 보이는 크기 계산
        display_width = image_width * current_scale
        display_height = image_height * current_scale
        display_diagonal = (display_width ** 2 + display_height ** 2) ** 0.5
        
        # 기준 화면 크기 (800x600 정도에서 브러시 크기 5가 적당)
        base_display_diagonal = (800 ** 2 + 600 ** 2) ** 0.5
        
        # 화면 표시 크기에 따른 스케일 팩터
        display_scale_factor = display_diagonal / base_display_diagonal
        
        # 기본 브러시 크기 (화면 기준)
        base_brush_size = 5.0
        base_slider_value = 60
        
        # 화면 크기에 맞춰 조정
        adjusted_brush_size = base_brush_size * display_scale_factor
        adjusted_slider_value = max(10, min(300, base_slider_value * display_scale_factor))
        
        # 실제 이미지 좌표계로 변환 (스케일 팩터 적용)
        image_brush_size = adjusted_brush_size / current_scale
        
        return image_brush_size, int(adjusted_slider_value)