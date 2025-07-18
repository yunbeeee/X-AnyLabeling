import copy
import math

from PyQt5 import QtCore, QtGui

from . import utils
from ..labeling.logger import logger

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)  # bf hovering
DEFAULT_FILL_COLOR = QtGui.QColor(100, 100, 100, 100)  # hovering
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)  # selected
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 255, 0, 155)  # selected
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 255)  # hovering
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 255, 255, 255)  # hovering


class Shape:
    """Shape data type"""

    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    KEYS = [
        "label",
        "score",
        "points",
        "group_id",
        "difficult",
        "shape_type",
        "flags",
        "description",
        "attributes",
    ]

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 4
    scale = 1.5
    line_width = 2.0

    def __init__(
        self,
        label=None,
        score=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        difficult=False,
        direction=0,
        attributes={},
        kie_linking=[],
        mask=None, ## modified: add mask type
    ):
        self.label = label
        self.score = score
        self.group_id = group_id
        self.description = description
        self.difficult = difficult
        self.kie_linking = kie_linking
        self.points = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type
        self.flags = flags
        self.other_data = {}
        self.attributes = attributes
        self.cache_label = None
        self.cache_description = None
        self.visible = True

        # Rotation setting
        self.direction = direction
        self.center = None
        self.show_degrees = True

        # Mask setting
        self.mask = mask

        self._highlight_index = None
        self._highlight_mode = self.NEAR_VERTEX
        self._highlight_settings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._vertex_fill_color = None

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color
        self.shape_type = shape_type

    def to_dict(self):
        dictData = {
            "label": self.label,
            "score": self.score,
            "points": [(p.x(), p.y()) for p in self.points],
            "group_id": self.group_id,
            "description": self.description,
            "difficult": self.difficult,
            "shape_type": self.shape_type,
            "flags": self.flags,
            "attributes": self.attributes,
            "kie_linking": self.kie_linking,
        }
        if self.shape_type == "rotation":
            dictData["direction"] = self.direction
        if self.shape_type == "mask" and self.mask is not None: # modified: 133-140
            import base64
            import numpy as np
            # Encode mask as base64 string
            mask_bytes = self.mask.tobytes()
            dictData["mask"] = base64.b64encode(mask_bytes).decode('utf-8')
            dictData["mask_shape"] = self.mask.shape
            dictData["mask_dtype"] = str(self.mask.dtype)
        dictData = {
            **self.other_data,
            **dictData,
        }
        return dictData

    def load_from_dict(self, data: dict, close=True):
        self.label = data["label"]
        self.score = data.get("score")
        self.points = [QtCore.QPointF(p[0], p[1]) for p in data["points"]]
        self.group_id = data.get("group_id")
        self.description = data.get("description", "")
        self.difficult = data.get("difficult", False)
        self.shape_type = data.get("shape_type", "polygon")
        self.flags = data.get("flags", {})
        self.attributes = data.get("attributes", {})
        self.kie_linking = data.get("kie_linking", [])
        if self.shape_type == "rotation":
            self.direction = data.get("direction", 0)
        if self.shape_type == "mask" and "mask" in data: # modified: 160-167
            import base64
            import numpy as np
            # Decode mask from base64 string
            mask_bytes = base64.b64decode(data["mask"])
            mask_shape = data["mask_shape"]
            mask_dtype = data["mask_dtype"]
            self.mask = np.frombuffer(mask_bytes, dtype=mask_dtype).reshape(mask_shape)
        self.other_data = {k: v for k, v in data.items() if k not in self.KEYS}
        if close:
            self.close()
        return self

    @property
    def shape_type(self):
        """Get shape type (polygon, rectangle, rotation, point, line, ...)"""
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        """Set shape type"""
        if value is None:
            value = "polygon"
        if value not in self.get_supported_shape():
            raise ValueError(f"Unexpected shape_type: {value}")
        self._shape_type = value

    @staticmethod
    def get_supported_shape():
        return [
            "polygon",
            "rectangle",
            "rotation",
            "point",
            "line",
            "circle",
            "linestrip",
            "mask", # modified:
        ]

    def close(self):
        """Close the shape"""
        if self.shape_type == "rotation" and len(self.points) == 4:
            cx = (self.points[0].x() + self.points[2].x()) / 2
            cy = (self.points[0].y() + self.points[2].y()) / 2
            self.center = QtCore.QPointF(cx, cy)
        self._closed = True

    def reach_max_points(self):
        if len(self.points) >= 4:
            return True
        return False

    def add_point(self, point):
        """Add a point"""
        if self.shape_type == "rectangle":
            if not self.reach_max_points():
                self.points.append(point)
        else:
            if self.points and point == self.points[0]:
                self.close()
            else:
                self.points.append(point)

    def can_add_point(self):
        """Check if shape supports more points"""
        return self.shape_type in ["polygon", "linestrip"]

    def pop_point(self):
        """Remove and return the last point of the shape"""
        if self.points:
            return self.points.pop()
        return None

    def insert_point(self, i, point):
        """Insert a point to a specific index"""
        self.points.insert(i, point)

    def remove_point(self, i):
        """Remove point from a specific index"""
        self.points.pop(i)

    def is_closed(self):
        """Check if the shape is closed"""
        return self._closed

    def set_open(self):
        """Set shape to open - (_close=False)"""
        self._closed = False

    def get_rect_from_line(self, pt1, pt2):
        """Get rectangle from diagonal line"""
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter: QtGui.QPainter):  # noqa: max-complexity: 18
        """Paint shape using QPainter"""
        if self.points:
            color = (
                self.select_line_color if self.selected else self.line_color
            )
            pen = QtGui.QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(self.line_width / self.scale))))
            painter.setPen(pen)

            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()

            if self.shape_type == "rectangle":
                assert len(self.points) in [1, 2, 4]
                if len(self.points) == 2:
                    rectangle = self.get_rect_from_line(*self.points)
                    line_path.addRect(rectangle)
                if len(self.points) == 4:
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.selected:
                            self.draw_vertex(vrtx_path, i)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
            elif self.shape_type == "rotation":
                assert len(self.points) in [1, 2, 4]
                if len(self.points) == 2:
                    rectangle = self.get_rect_from_line(*self.points)
                    line_path.addRect(rectangle)
                if len(self.points) == 4:
                    line_path.moveTo(self.points[0])
                    for i, p in enumerate(self.points):
                        line_path.lineTo(p)
                        if self.selected:
                            self.draw_vertex(vrtx_path, i)
                    if self.is_closed() or self.label is not None:
                        line_path.lineTo(self.points[0])
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.get_circle_rect_from_line(self.points)
                    line_path.addEllipse(rectangle)
                if self.selected:
                    for i in range(len(self.points)):
                        self.draw_vertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if self.selected:
                        self.draw_vertex(vrtx_path, i)
            elif self.shape_type == "point":
                assert len(self.points) == 1
                self.draw_vertex(vrtx_path, 0, True)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                self.draw_vertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    if self.selected:
                        self.draw_vertex(vrtx_path, i)
                if self.is_closed():
                    line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            if self._vertex_fill_color is not None:
                painter.fillPath(vrtx_path, self._vertex_fill_color)
            if self.fill:
                color = (
                    self.select_fill_color
                    if self.selected
                    else self.fill_color
                )
                painter.fillPath(line_path, color)

    def draw_vertex(self, path, i, show_difficult=False):
        """Draw a vertex"""
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlight_index:
            size, shape = self._highlight_settings[self._highlight_mode]
            d *= size
        if self._highlight_index is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color
        if shape in (self.P_SQUARE, self.P_ROUND):
            if self.difficult and show_difficult:
                scale_factor = 1.5
                triangle_path = QtGui.QPainterPath()
                triangle_path.moveTo(
                    point.x(), point.y() - d * scale_factor / 2
                )
                triangle_path.lineTo(
                    point.x() - d * scale_factor / 2,
                    point.y() + d * scale_factor / 2,
                )
                triangle_path.lineTo(
                    point.x() + d * scale_factor / 2,
                    point.y() + d * scale_factor / 2,
                )
                triangle_path.closeSubpath()
                path.addPath(triangle_path)
                if shape == self.P_ROUND:
                    path.addPath(triangle_path)
            else:
                if shape == self.P_SQUARE:
                    path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
                elif shape == self.P_ROUND:
                    path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            logger.error("Unsupported vertex shape")

    def nearest_vertex(self, point, epsilon):
        """Find the index of the nearest vertex to a point
        Only consider if the distance is smaller than epsilon
        """
        min_distance = float("inf")
        min_i = None
        for i, p in enumerate(self.points):
            dist = utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearest_edge(self, point, epsilon):
        """Get nearest edge index"""
        min_distance = float("inf")
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = utils.distance_to_line(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def contains_point(self, point):
        """Check if shape contains a point"""
        if self.is_mask() and self.mask is not None:
            # print("[DEBUG] contains_point: mask is not None")
            x, y = int(point.x()), int(point.y())
            if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
                return self.mask[y, x] > 0
            return False
        return self.make_path().contains(point)

    def get_circle_rect_from_line(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, _) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def make_path(self):
        """Create a path from shape"""
        if self.is_mask():
            return QtGui.QPainterPath()
        if self.shape_type == "rectangle":
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.get_circle_rect_from_line(self.points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def bounding_rect(self):
        """Return bounding rectangle of the shape"""
        if self.is_mask() and self.mask is not None:
            import numpy as np
            ys, xs = np.nonzero(self.mask)
            if len(xs) == 0 or len(ys) == 0:
                return QtCore.QRectF()
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            return QtCore.QRectF(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        return self.make_path().boundingRect()

    def move_by(self, offset):
        """Move all points by an offset"""
        self.points = [p + offset for p in self.points]

    def move_vertex_by(self, i, offset):
        """Move a specific vertex by an offset"""
        self.points[i] = self.points[i] + offset

    def highlight_vertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlight_index = i
        self._highlight_mode = action

    def highlight_clear(self):
        """Clear the highlighted point"""
        self._highlight_index = None

    def copy(self):
        """Copy shape"""
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

    # modified: 475-486
    def is_mask(self):
        """Check if shape is a mask type"""
        return self.shape_type == "mask"

    def get_mask(self):
        """Get mask data"""
        return self.mask if self.is_mask() else None

    def set_mask(self, mask):
        """Set mask data"""
        if self.is_mask():
            self.mask = mask
