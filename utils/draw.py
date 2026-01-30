import math
from enum import Enum
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class Color(Enum):
    RED = (75, 25, 230)
    GREEN = (75, 180, 60)
    YELLOW = (0, 225, 255)
    BLUE = (200, 130, 0)
    ORANGE = (48, 130, 245)
    PURPLE = (180, 30, 145)
    CYAN = (240, 240, 70)
    MAGENTA = (230, 50, 240)
    LIME = (60, 245, 210)
    PINK = (212, 190, 250)
    TEAL = (128, 128, 0)
    LAVENDER = (255, 190, 220)
    BROWN = (40, 110, 170)
    BEIGE = (200, 250, 255)
    MAROON = (0, 0, 128)
    MINT = (195, 255, 170)
    OLIVE = (0, 128, 128)
    APRICOT = (180, 215, 255)
    NAVY = (128, 0, 0)
    GREY = (128, 128, 128)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    HOTPINK = (180, 105, 255)  # (B, G, R) from 255, 105, 180
    DEEPPINK = (147, 20, 255)  # (B, G, R) from 255, 20, 147
    FUCHSIA = (255, 0, 255)  # (B, G, R) from 255, 0, 255
    LEMON = (205, 250, 255)  # (B, G, R) from 255, 250, 205

    def __call__(self):
        return self.value

    def __str__(self):
        return f"{self.name.lower()} = {self.value}"


def get_color(color_name: str) -> Color:
    try:
        return Color[color_name.upper()].value
    except KeyError:
        raise ValueError(f"{color_name} is not a valid color name")


def draw_circle(
    mat: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws a circle on the input image. The input image is modified in-place.

    Args:
        mat: input image
        center: center of the circle
        radius: radius of the circle
        color: color of the circle, in BGR.
        thickness: thickness of circle boundary; if negative, a
            filled circle is drawn

    Returns:
        None.
    """
    cv2.circle(mat, center, radius, color, thickness=thickness)


def draw_ellipse(
    mat: np.ndarray,
    radius_x: int,
    radius_y: int,
    angle: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws an ellipse on the input image. The input image is modified in-place.

    Args:
        mat: input image
        radius_x: radius in the x-axis
        radius_y: radius in the y-axis
        angle: rotation angle of the ellipse, in degrees
        color: color of the ellipse, in BGR
        thickness: thickness of ellipse boundary; if negative, a filled
            ellipse is drawn

    Returns:
        None.
    """
    cv2.ellipse(mat, (radius_x, radius_y), angle, 0, 0, color, thickness=thickness)


def draw_line(
    mat: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws a line on the input image. The input image is modified in-place.

    Args:
        mat: input image.
        pt1: first point on the line.
        pt2: second point on the line.
        color: color of the line, in BGR
        thickness: thickness of the line.

    Returns:
        None.
    """
    cv2.line(mat, pt1, pt2, color, thickness=thickness)


def draw_arrow(
    mat: np.ndarray,
    from_pt: Tuple[int, int],
    to_pt: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws an arrow on the input image. The input image is modified in-place.

    Args:
        mat: input image
        from_pt: point at the base of the arrow
        to_pt: point at the tip of the arrow
        color: color of the arrow, in BGR
        thickness: thickness of the arrow

    Returns:
        None.
    """
    cv2.arrowedLine(mat, from_pt, to_pt, color, thickness=thickness)


def draw_rect(
    mat: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws a rectangle on the input image. The input image is modified in-place.

    Args:
        mat: input image
        pt1: vertex of the rectangle
        pt2: vertex of the rectangle opposite from pt1
        color: color of the rectangle
        thickness: thickness of the borders of the rectangle. if negative,
            a filled rectangle is drawn

    Returns:
        None.
    """
    cv2.rectangle(mat, pt1, pt2, color, thickness=thickness)


def draw_rot_rect(
    mat: np.ndarray,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    angle: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws a rotated rectangle on the input image. The input image is modified
    in-place.

    Args:
        mat: input image
        center_x: x-coordinate of the center of the rectangle
        center_y: y-coordinate of the center of the rectangle
        width: width of the rectangle
        height: height of the rectangle
        angle: rotation angle of the rectangle, in degrees
        color: color of the rectangle, in BGR
        thickness: thickness of the borders of the rectangle

    Returns:
        None.
    """
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (
        int(center_x - a * height - b * width),
        int(center_y + b * height - a * width),
    )
    pt1 = (
        int(center_x + a * height - b * width),
        int(center_y - b * height - a * width),
    )
    pt2 = (int(2 * center_x - pt0[0]), int(2 * center_y - pt0[1]))
    pt3 = (int(2 * center_x - pt1[0]), int(2 * center_y - pt1[1]))

    cv2.line(mat, pt0, pt1, color, thickness)
    cv2.line(mat, pt1, pt2, color, thickness)
    cv2.line(mat, pt2, pt3, color, thickness)
    cv2.line(mat, pt3, pt0, color, thickness)


def draw_text(
    mat: np.ndarray,
    s: str,
    origin: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws text on the input image. The input image is modified in-place.

    Args:
        mat: input image
        s: text to draw
        origin: coordinate of bottom-left corner of text
        scale: font scaling factor
        color: color of font, in BGR
        thickness: thickness of font

    Returns:
        None.
    """
    cv2.putText(
        mat, s, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness=thickness
    )


def draw_text_with_background(
    mat: np.ndarray,
    s: str,
    origin: Tuple[int, int],
    scale: float,
    text_color: Tuple[int, int, int] = (0, 0, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 3,
) -> None:
    """
    Draws text on the input image with a solid rectangle background. The input image is modified in-place.

    Args:
        mat: Input image.
        s: Text to draw.
        origin: Coordinate of bottom-left corner of text.
        scale: Font scaling factor.
        text_color: Color of the font in BGR.
        background_color: Color of the background rectangle in BGR.
        thickness: Thickness of the font.
        padding: Padding around the text inside the rectangle.

    Returns:
        None.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(s, font, scale, thickness)
    top_left = (origin[0] - padding, origin[1] - text_height - padding)
    bottom_right = (origin[0] + text_width + padding, origin[1] + baseline + padding)

    # Draw the rectangle background
    cv2.rectangle(mat, top_left, bottom_right, background_color, -1)

    # Draw the text on top of the rectangle
    cv2.putText(mat, s, origin, font, scale, text_color, thickness=thickness)


def draw_contours(
    mat: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws contours on the input image. The input image is modified in-place.

    Args:
        mat: input image
        contours: contours to draw
        color: color of contours, in BGR
        thickness: thickness of contours; filled if -1

    Returns:
        None.
    """
    cv2.drawContours(mat, contours, -1, color, thickness=thickness)


def draw_polylines(
    mat: np.ndarray,
    points: List[Tuple[int, int]],
    isClosed: bool = False,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """
    Draws a polyline (or polygon) on the input image. The input image is modified in-place.

    Args:
        mat: input image
        points: list of points representing the polyline
        is_closed: whether the polyline is closed (forming a polygon)
        color: color of the polyline, in BGR
        thickness: thickness of the polyline

    Returns:
        None.
    """
    points_array = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        mat, [points_array], isClosed=isClosed, color=color, thickness=thickness
    )
