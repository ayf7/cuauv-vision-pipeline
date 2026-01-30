import cv2
import numpy as np

def to_odd_linear(n:int) -> int:
    """
    Converts a number to an odd number using a linear function.

    Args:
        n: the number

    Returns:
        odd number by applying a linear function.
    """
    return 2 * n + 1


def to_odd(n:int) -> int:
    """
    Converts a number to an odd number by getting the nearest odd number
    (greater than or equal to the input)

    Args:
        n: the number

    Returns:
        projected odd number
    """
    return n if n % 2 == 1 else n + 1


def to_umat(mat:cv2.Mat) -> cv2.UMat:
    """
    Converts an image matrix to a unified matrix in the Transparent API for transparent access to
    both CPU/OpenCL-accelerated codepaths.

    Args:
        mat: Mat-like image.
    
    Returns:
        UMat from image.
    """
    return cv2.UMat(mat)

 
def from_umat(umat: cv2.UMat) -> cv2.Mat:
    """
    Converts a unified image matrix to a standard image matrix.
    
    Args:
        umat: the cv2.UMat image.
    
    Returns:
        standard matrix image.
    """
    return cv2.UMat.get(umat)


def as_mat(mat: cv2.UMat | cv2.Mat) -> cv2.Mat:
    """
    Converts an image matrix (either in standard form or unified form) to a standard image matrix
    
    Args:
        mat: standard image or unified image matrix form.
    
    Returns:
        standard matrix image.
    """
    return from_umat(mat) if isinstance(mat, cv2.UMat) else mat

