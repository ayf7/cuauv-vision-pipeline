import cv2
import numpy as np
from typing import List, Tuple

def outer_contours(mat: np.ndarray) -> List[np.ndarray]:
    """
    Extracts all outermost contours in the image.

    Contours are continuous curves that bound regions of the same intensity. 
    This function specifically finds the outermost contours (those not enclosed 
    by other contours), which can be useful for object detection.

    Args:
        mat: Input image in grayscale. 
            Preprocessing like thresholding or edge detection may be required.

    Returns:
        Detected outermost contours.
    """
    contours, _ = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def all_contours(mat: np.ndarray) -> List[np.ndarray]:
    """
    Extracts all contours in the image, including nested and inner contours.

    Contours are continuous curves that bound regions of the same intensity. 
    This function extracts all contours, not just the outermost ones, allowing 
    detection of structures within other objects.

    Args:
        mat: Input image in grayscale.

    Returns:
        All contours, including those that are enclosed by others.
    """
    contours, _ = cv2.findContours(mat, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def canny(mat: np.ndarray,
          lower: int,
          upper: int) -> np.ndarray:
    """
    Performs Canny edge detection on the image using the given thresholds for
    hysteresis.

    Canny edge detection identifies edges in an image via a sequence of Gaussian
    noise filtering, calculation of an intensity gradient over the image,
    filtering out any pixels which do not lie on an edge (are not a local
    maximum in the direction of the gradient), and finally identifying edges via
    a dual-threshold hysteresis on the intensity gradient values. Any pixels
    connected to pixels that are above the high threshold are assumed to be part
    of the edge. Any pixels below the low threshold are discarded.

    Args:
        mat: Input image.
        lower: Lower gradient threshold, below which all edges are discarded.
        upper: Upper gradient threshold, above which all edges are accepted.

    Returns:
        Binary edge image after applying Canny edge detection.
    """
    return cv2.Canny(mat, lower, upper)


def simple_canny(mat: np.ndarray,
                 sigma: float = 0.33,
                 use_mean: bool = False) -> np.ndarray:
    """
    Performs Canny edge detection on the image using automatically computed
    lower and upper thresholds for hysteresis.

    Canny edge detection identifies edges in an image via a sequence of Gaussian
    noise filtering, calculation of an intensity gradient over the image,
    filtering out any pixels which do not lie on an edge (are not a local
    maximum in the direction of the gradient), and finally identifying edges via
    a dual-threshold hysteresis on the intensity gradient values. Any pixels
    connected to pixels that are above the high threshold are assumed to be part
    of the edge. Any pixels below the low threshold are discarded.

    Source:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

        
    Args:
        mat: Input image.
        sigma: Deviation from median (or mean) used to compute thresholds.
            Defaults to 0.33.
        use_mean: Whether to calculate deviation from arithmetic mean instead
            of median. Defaults to False.

    Returns:
        Binary edge image after applying Canny edge detection.
    """
    mid = np.mean(mat) if use_mean else np.median(mat)
    lower = int(max(0, (1.0 - sigma) * mid))
    upper = int(min(255, (1.0 + sigma) * mid))
    return cv2.Canny(mat, lower, upper)


def find_corners(mat: np.ndarray,
                 max_corners: int,
                 quality_thresh: float = 0.01,
                 min_distance: int = 10) -> np.ndarray:
    """
    Detects corners in the image using the Shi-Tomasi method, which is a variant
    of the Harris Corner Detector. The Harris Corner Detector identifies corners
    by locating rectangular subregions of an image that would introduce large
    changes into the image when moved. Corners are considered good features for
    tracking, because corners are invariant to translation, rotation, and
    illumination.

    Args:
        mat: Input image in grayscale.
        max_corners: Maximum number of corners to detect.
        quality_thresh: Minimum quality threshold for corners. Defaults to 0.01.
        min_distance: Minimum distance between detected corners. Defaults to 10.

    Returns:
        An array of detected corners.
    """
    return cv2.goodFeaturesToTrack(mat, max_corners, quality_thresh, min_distance)


def find_circles(mat: np.ndarray,
                 res_ratio: float,
                 min_distance: int,
                 canny_thresh: int = 100, 
                 circle_thresh: int = 100,
                 min_radius: int = 0,
                 max_radius: int = 0) -> np.ndarray:
    """
    Find circles using the circle Hough Transform. The circle Hough Transform
    detects circles by voting for different coordinates within the Hough
    parameter space ((x, y, r) for a circle center and radius) and identifying
    local maxima in the accumulator.

    Args:
        mat: Input image; it is recommended to apply noise reduction beforehand.
        res_ratio: Inverse ratio of accumulator resolution to image resolution.
        min_distance: Minimum distance between circle centers.
        canny_thresh: Value used to scale upper and lower thresholds for Canny
            detector. Defaults to 100.
        circle_thresh: Accumulator threshold for circle centers. Defaults to 100.
        min_radius: Minimum radius of circles. Defaults to 0.
        max_radius: Maximum radius of circles. Defaults to 0.

    Returns:
        An array of detected circles, each represented as (x, y, radius).
    """
    return cv2.HoughCircles(mat, cv2.HOUGH_GRADIENT, res_ratio, min_distance, param1=canny_thresh, param2=circle_thresh,
                            minRadius=min_radius, maxRadius=max_radius)


def line_polar_to_cartesian(rho: float,
                            theta: float) -> Tuple[int, int, int, int]:
    """
    Converts a line in polar coordinates to a pair of points in cartesian
    coordinates.

    Args:
        rho: Distance from the origin to the line.
        theta: Angle in radians between the x-axis and the line.

    Returns:
        A tuple (x1, y1, x2, y2) corresponding to two points (x1, y1) and
        (x2, y2) on the line.
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)
    return (x1, y1, x2, y2)


def find_lines(mat: np.ndarray,
               rho: float,
               theta: float,
               threshold: int) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[float, float]]]:
    """
    Finds lines using the standard Hough Transform.  The standard Hough
    Transform detects lines by voting for different coordinates within the Hough
    parameter space ((r, theta) for the polar coordinates of the line) and
    identifying local maxima in the accumulator.

    Args:
        mat: Input image.
        rho: Distance resolution of the accumulator (in pixels).
        theta: Angle resolution of the accumulator (in radians).
        threshold: Accumulator threshold for lines.

    Returns:
        A tuple containing two lists - first, A list of (x1, y1, x2, y2)
        representing line segments, and second, a list of (r, theta)
        representing lines in polar coordinates.
    """
    cartesian_points = []
    polar_points = []
    lines = cv2.HoughLines(mat, rho, theta, threshold)
    if lines is not None:
        for line in lines:
            (rho, theta) = line[0]
            cartesian_points.append(line_polar_to_cartesian(rho, theta))
            polar_points.append((rho, theta))
    return cartesian_points, polar_points


def find_line_segments(mat: np.ndarray,
                       rho: float,
                       theta: float,
                       threshold: int,
                       min_line_length: int = 0,
                       max_line_gap: int = 0) -> np.ndarray:
    """
    Finds line segments using the Probabilistic Hough Transform, a more
    efficient implementation of the standard Hough Transform.

    Args:
        mat: Input image.
        rho: Distance resolution of the accumulator (in pixels).
        theta: Angle resolution of the accumulator (in radians).
        threshold: Accumulator threshold for lines.
        min_line_length: Minimum line length. Defaults to 0.
        max_line_gap: Maximum allowed gap between points on the same
            line. Defaults to 0.

    Returns:
        An array of detected line segments, each represented by (x1, y1, x2, y2).
    """
    return cv2.HoughLinesP(mat, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)


def contour_centroid(contour: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the centroid (center of mass) of the contour.

    Args:
        contour: Input contour.

    Returns:
        A tuple (x, y) representing the centroid coordinates.
    """
    moments = cv2.moments(contour)
    m00 = max(1e-10, moments['m00'])
    return int(moments['m10'] / m00), int(moments['m01'] / m00)


def contour_area(contour: np.ndarray) -> float:
    """
    Calculates the area of the contour.

    Args:
        contour: Input contour.

    Returns:
        The area of the contour.
    """
    return cv2.contourArea(contour, oriented=False)


def contour_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """
    Calculates the perimeter of the contour.

    Args:
        contour: Input contour.
        closed: Whether the contour is closed. Defaults to True.

    Returns:
        The perimeter of the contour if it is closed; otherwise, the arclength
        of the contour.
    """
    return cv2.arcLength(contour, closed)


def contour_approx(contour: np.ndarray, epsilon: float = None, closed: bool = True) -> np.ndarray:
    """
    Returns an approximation of the given contour using fewer vertices.

    Args:
        contour: Input contour.
        epsilon: Maximum distance between approximated contour and original contour.
            If None, it's set to 0.1 * contour_perimeter(contour, closed).
        closed: Whether the contour is closed. Defaults to True.

    Returns:
        The approximated contour.
    """
    if epsilon is None:
        epsilon = 0.1 * contour_perimeter(contour, closed)
    return cv2.approxPolyDP(contour, epsilon, closed)


def min_enclosing_rect(contour: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Returns a minimum area enclosing rectangle of the contour.

    Args:
        contour: Input contour.

    Returns:
        A tuple ((center_x, center_y), (width, height), rotation_angle) of the rectangle.
        The rotation angle is in degrees.
    """
    return cv2.minAreaRect(contour)


def min_enclosing_circle(contour: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    Returns a minimum area enclosing circle of the contour.

    Args:
        contour: Input contour.

    Returns:
        A tuple ((center_x, center_y), radius) of the circle.
    """
    return cv2.minEnclosingCircle(contour)


def min_enclosing_ellipse(contour: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Returns an ellipse in which the contour is inscribed.

    The ellipse is fitted to the contour using a least-squares approximation,
    so the contour may not be entirely enclosed.

    Args:
        contour: Input contour.

    Returns:
        A tuple ((center_x, center_y), (radius_x, radius_y), rotation_angle) of
        the ellipse. The rotation angle is in degrees.
    """
    return cv2.fitEllipse(contour)

