import cv2
import numpy as np
from typing import Optional

def simple_gaussian_blur(mat: np.ndarray, kernel_size: int, std_dev: float) -> np.ndarray:
    """
    Blurs an image using a Gaussian filter with a square kernel.

    Args:
        mat:  Input image. Can be 2D grayscale or 3D color array.
        kernel_size: Size of the square convolution kernel. Must be odd.
        std_dev: Standard deviation of the Gaussian distribution
            for both X and Y axes.

    Returns:
        out: A numpy array of the same shape and type
            as the input, containing the blurred image.

    Raises:
        ValueError (ValueError): If kernel_size is not an odd integer.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer")
    
    return cv2.GaussianBlur(mat, (kernel_size, kernel_size), std_dev)

def elliptic_kernel(x: int, y: Optional[int] = None) -> np.ndarray:
    """Returns an elliptic kernel for use in morphological transforms.

    Args:
        x: Length of the x-axis of the ellipse. Must be an odd
            positive integer.
        y: Length of the y-axis of the ellipse. Must
            be an odd positive integer. If None, it's set equal to x,
            creating a circular kernel.

    Returns:
        out: A 2D boolean numpy array representing the
            elliptic kernel.

    Raises:
        ValueError (ValueError): If x or y is not an odd positive
            integer.
    """
    if y is None:
        y = x
    
    if x % 2 == 0 or y % 2 == 0 or x <= 0 or y <= 0:
        raise ValueError("x and y must be odd positive integers")

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, y))


def rect_kernel(x: int, y: Optional[int] = None) -> np.ndarray:
    """Returns a rectangular kernel for use in morphological transforms.

    Args:
        x: Length of the x-axis (width) of the rectangle. Must be
            a positive integer.
        y: Length of the y-axis (height) of the rectangle.
            Must be a positive integer. If None, it's set equal to x,
            creating a square kernel.

    Returns:
        out: 2D boolean numpy array representing the
            rectangular kernel.

    Raises:
        ValueError (ValueError): If x or y is not a positive integer.
    """
    if y is None:
        y = x
    
    if x <= 0 or y <= 0:
        raise ValueError("x and y must be positive integers")

    return cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))


def erode(mat: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Applies erosion to the input image, which reduces the boundaries of the
    foreground objects. 

    Args:
        mat: Input image, typically a binary image where
            foreground is white (255) and background is black (0).
        kernel: Kernel used for erosion.
        iterations: Number of times erosion is applied. Defaults to 1.

    Returns:
        Eroded image of the same shape and type as the input.
    """
    return cv2.erode(mat, kernel, iterations=iterations)


def dilate(mat: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Applies dilation to the input image, which expands the boundaries of
    foreground objects. This operation is useful for closing small black holes
    inside white regions or connecting small disjoint white regions.

    Args:
        mat: Input image, typically a binary image where
            foreground is white (255) and background is black (0).
        kernel: Kernel used for dilation.
        iterations: Number of times dilation is applied. Defaults to 1.

    Returns:
        Dilated image of the same shape and type as the input.
    """
    return cv2.dilate(mat, kernel, iterations=iterations)


def morph_remove_noise(mat: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Applies an opening operation (erosion followed by dilation) to remove small
    white noise from the input image. This is often used to clean up binary
    images where noise pixels appear as small foreground objects.

    Args:
        mat: Input binary image.
        kernel: Kernel used for the opening operation.
        iterations: Number of times operation is applied.

    Returns:
        Image after noise removal.
    """
    return cv2.morphologyEx(mat, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close_holes(mat: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Applies a closing operation (dilation followed by erosion) to close small
    black holes in the foreground of the input image. This is often used to fill
    small gaps or holes within objects.

    Args:
        mat: Input binary image.
        kernel: Kernel used for the closing operation.
        iterations: Number of times the operation is applied. Defaults to 1.
    
    Returns:
        Image after hole closing.
    """
    return cv2.morphologyEx(mat, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morph_borders(mat: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Computes the morphological gradient of the input image by finding the
    difference between the dilation and erosion of the image. This highlights
    the boundaries of the objects in the image.

    Args:
        mat: Input binary image.
        kernel: Kernel used for the morphological gradient.
        iterations: Number of times the morphological gradient operation is 
            applied. Defaults to 1.

    Returns:
        Image showing the borders of objects.
    """
    return cv2.morphologyEx(mat, cv2.MORPH_GRADIENT, kernel, iterations=iterations)


def resize(mat: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resizes the input image to the specified width and height.

    Args:
        mat: Input image.
        width: New width of the image.
        height: New height of the image.

    Returns:
        Resized image.
    """
    return cv2.resize(mat, (width, height))


def rotate(mat: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotates the input image by the specified number of degrees about its center.
    The border pixels are extrapolated by replication to avoid missing data at
    the corners.

    Args:
        mat: Input image.
        degrees: Number of degrees to rotate the image. Positive values
            rotate counterclockwise.

    Returns:
        Rotated image.
    """
    rot_mat = cv2.getRotationMatrix2D((mat.shape[1] / 2, mat.shape[0] / 2), degrees, 1)
    return cv2.warpAffine(mat, rot_mat, (mat.shape[1], mat.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def translate(mat: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Translates the input image by the specified amounts in the x and y
    directions.

    Args:
        mat: Input image.
        x: Amount to translate the image along the x-axis.
        y: Amount to translate the image along the y-axis.

    Returns:
        Translated image.
    """
    trans_mat = np.float32([[1, 0, x],
                            [0, 1, y]])
    return cv2.warpAffine(mat, trans_mat, (mat.shape[1], mat.shape[0]))

def decode_normal(mat: np.ndarray) -> np.ndarray:
    """
    Converts a normal map from [0, 255] integer values to floating point values
    in the range [-1, 1]. The normal map in [0, 255] is utilized for
    compatibility with the WebGUI, but we then convert it back to its normal
    vectors. The norm of mat[i][j] is approximately 1.

    Args:
        mat: Input normal map, with pixel values in the range [0, 255].

    Returns:
        Decoded normal map with values in the range [-1, 1].
    """
    img = mat.astype(np.float32)
    img = (img / 255.0) * 2.0 - 1.0 # reversing transformations from [0, 255] -> [-1, 1].
    return img