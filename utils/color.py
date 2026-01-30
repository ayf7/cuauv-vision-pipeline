from math import sqrt
import ctypes
import cv2
import numpy as np

from auv_python_helpers import load_library
from vision.utils.helpers import as_mat
from typing import Callable, Tuple, List


def _convert_colorspace(conv_type: int) -> Callable[[np.ndarray], Tuple[np.ndarray, List[np.ndarray]]]:
    """Function generator to create colorspace conversion of an image.
    
    Args:
        conv_type: The type of colorspace conversion.

    Returns:
        A function that takes an input image and returns (converted image, target colorspace split).
    """
    def _inner(mat: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        conv = cv2.cvtColor(mat, conv_type)
        return conv, cv2.split(conv)
    return _inner


_conversions = [cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2YCrCb,
                cv2.COLOR_BGR2LUV, cv2.COLOR_BGR2GRAY, cv2.COLOR_GRAY2BGR, cv2.COLOR_LAB2BGR,
                cv2.COLOR_HSV2BGR]

bgr_to_lab, bgr_to_hsv, bgr_to_hls, \
bgr_to_ycrcb, bgr_to_luv, bgr_to_gray, \
gray_to_bgr, lab_to_bgr, hsv_to_bgr = [_convert_colorspace(c) for c in _conversions]


def color_dist(c1: Tuple[int, int, int],
               c2: Tuple[int, int, int]) -> float:
    """
    Calculates the Euclidean distance between two colors.

    Args:
        c1: First color (3-dimensional).
        c2: Second color (3-dimensional).

    Returns:
        The Euclidean distance between color values.
    """
    return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def elementwise_color_dist(mat: np.ndarray,
                           c: Tuple[int, int, int]) -> np.ndarray:
    """
    Calculates the elementwise distance between the pixels in an image matrix
    and a given color.

    Args:
        mat: Image matrix.
        c: Target color.

    Returns:
        A matrix of distance values for each pixel in the image.
    """
    return np.linalg.norm(as_mat(mat) - c, axis=2)


def thresh_color_distance(split: List[np.ndarray],
                          color: Tuple[int, int, int],
                          distance: float, 
                          auto_distance_percentile: float = None,
                          ignore_channels: List[int] = [], 
                          weights: Tuple[float, float, float] = (1, 1, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thresholds the image according to the weighted distance of each pixel to the
    color.

    Args:
        split: A list of monocolored images (see _convert_colorspace).
        color: The target color of the threshold.
        distance: The target distance of the threshold.
        auto_distance_percentile: The percentile used for distance.
        ignore_channels: Specifies which color channels should be ignored.
        weights: The weights to calculate the weighted color distance of each channel.

    Returns:
        A tuple of the input image with all pixel values with weighted color
        distance to color larger than distance set to zero and all other values
        set to 255, and a matrix of distances of the point to the specified
        color.
    """
    weights_cp = list(weights)
    for idx in ignore_channels:
        weights_cp[idx] = 0
    weights_cp /= np.linalg.norm(weights)
    dists = np.zeros(split[0].shape, dtype=np.float32)
    for i in range(3):
        if i in ignore_channels:
            continue
        dists += weights_cp[i] * (np.float32(split[i]) - color[i])**2
    if auto_distance_percentile:
        distance = min(np.percentile(dists, auto_distance_percentile), distance**2)
    else:
        distance = distance**2
    return range_threshold(dists, 0, distance), np.uint8(np.sqrt(dists))

def range_threshold(mat: np.ndarray,
                    min: int,
                    max: int) -> np.ndarray:
    """
    Thresholds the image based on a range of values. The input image with all
    pixel values outside of the range [min, max] are set to zero  and all values
    within the range set to 255.

    Args:
        mat: Input image.
        min: Minimum threshold for pixel value.
        max: Maximum threshold for pixel value.

    Returns:
        Thresholded image.
    """
    return cv2.inRange(mat, min, max)


def binary_threshold(mat: np.ndarray,
                     threshold: int) -> np.ndarray:
    """
    Applies binary thresholding to the image. The values above the threshold are
    set to 255 and values less than or equal to the threshold are set to zero.

    Args:
        mat: Input image.
        threshold: Threshold value.

    Returns:
        Thresholded image.
    """
    return cv2.threshold(mat, threshold, 255, cv2.THRESH_BINARY)[1]


def binary_threshold_inv(mat: np.ndarray,
                         threshold: int) -> np.ndarray:
    """Applies inverse binary thresholding to the image. The values above the
    threshold are set to zero and values less than or equal to the threshold are
    set to 255.

    Args:
        mat: Input image.
        threshold: Threshold value.

    Returns:
        Thresholded image.
    """
    return cv2.threshold(mat, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def max_threshold(mat: np.ndarray,
                  threshold: float) -> np.ndarray:
    """
    Applies maximum thresholding to the image. The values above the threshold
    are set to threshold and all other values are unchanged.

    Args:
        mat: Input image. threshold:
        Threshold value.

    Returns:
        Thresholded image.
    """
    return cv2.threshold(mat, threshold, 0, cv2.THRESH_TRUNC)[1]


def above_threshold(mat: np.ndarray,
                    threshold: float) -> np.ndarray:
    """
    Thresholds the image, keeping values above the threshold.

    Args:
        mat: Input image.
        threshold: Threshold value.

    Returns:
        Image where values above the threshold are unchanged, and values less
        than or equal to the threshold are set to zero.
    """
    return cv2.threshold(mat, threshold, 0, cv2.THRESH_TOZERO)[1]


def below_threshold(mat: np.ndarray,
                    threshold: float) -> np.ndarray:
    """
    Thresholds the image, keeping values below the threshold.

    Args:
        mat: Input image.
        threshold: Threshold value.

    Returns:
        Image where values above the threshold are set to zero, and values less
        than or equal to the threshold are unchanged.
    """
    return cv2.threshold(mat, threshold, 0, cv2.THRESH_TOZERO_INV)[1]


def otsu_threshold(mat: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Applies Otsu's thresholding method to the image.

    The input image must be bimodal, because Otsu's Binarization method
    automatically calculates a threshold value based on the bimodal values.

    Args:
        mat: Input image (should be bimodal).

    Returns:
        A tuple containing the threshold value and the thresholded image.
    """
    return cv2.threshold(mat, 0, 255, cv2.THRESH_OTSU)


def adaptive_threshold_mean(mat: np.ndarray,
                            neighborhood_size: int,
                            bias: float = 0) -> np.ndarray:
    """
    Applies adaptive thresholding using the mean of neighboring values.

    Args:
        mat: Input image.
        neighborhood_size: Size of neighborhood for threshold calculation (must be odd).
        bias: A constant offset to the calculated mean value.

    Returns:
        The thresholded image where values above the threshold are set to 255 
        and values less than or equal to the threshold are set to zero.
    """
    return cv2.adaptiveThreshold(mat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, neighborhood_size, bias)


def adaptive_threshold_mean_inv(mat: np.ndarray,
                                neighborhood_size: int,
                                bias: float = 0) -> np.ndarray:
    """
    Applies inverse adaptive thresholding using the mean of neighboring
    values.

    Args:
        mat: Input image.
        neighborhood_size: Size of neighborhood for threshold calculation (must be odd).
        bias: A constant offset to the calculated mean value.

    Returns:
        The thresholded image where values above the threshold are set to zero 
        and values less than or equal to the threshold are set to 255.
    """
    return cv2.adaptiveThreshold(mat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, neighborhood_size, bias)


def adaptive_threshold_gaussian(mat: np.ndarray,
                                neighborhood_size: int,
                                bias: float = 0) -> np.ndarray:
    """
    Applies adaptive thresholding using a Gaussian-weighted sum of neighboring
    values.

    Args:
        mat: Input image.
        neighborhood_size: Size of neighborhood for threshold calculation (must be odd).
        bias: A constant offset to the calculated weighted mean value.

    Returns:
        The thresholded image where values above the threshold are set to 255 
        and values less than or equal to the threshold are set to zero.
    """
    return cv2.adaptiveThreshold(mat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_size, bias)


def adaptive_threshold_gaussian_inv(mat: np.ndarray,
                                    neighborhood_size: int,
                                    bias: float = 0) -> np.ndarray:
    """
    Applies inverse adaptive thresholding using a Gaussian-weighted sum of
    neighboring values.

    Args:
        mat: Input image.
        neighborhood_size: Size of neighborhood for threshold calculation (must be odd).
        bias: A constant offset to the calculated weighted mean value.

    Returns:
        The thresholded image where values above the threshold are set to zero 
        and values less than or equal to the threshold are set to 255.
    """
    return cv2.adaptiveThreshold(mat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neighborhood_size, bias)


def kmeans(mat: np.ndarray,
           num_centeroids: int,
           iterations: int = 10,
           epsilon: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs k-means clustering on the colors in an image.

    It clusters the colors as three-dimensional points and separates them into
    different clusteroids.

    Args:
        mat: The image to be clustered.
        num_centeroids: The number of clusters/centeroids to be used when
            running the algorithm.
        iterations: The number of iterations to run the kmeans algorithm.
        epsilon: The epsilon to terminate the algorithm when it is reached.

    Returns:
        A tuple containing: A measure of how close the samples are to each
        center, the best suited label for each point on the image, and the
        centeroids of each label.
    """
    array = mat.reshape((-1, 2))
    array = np.float32(array)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(array, num_centeroids, None, criteria, cv2.KMEANS_ITER, cv2.KMEANS_RANDOM_CENTERS)
    compactness =  np.reshape((mat.shape[0], mat.shape[1]))
    labels = np.reshape((mat.shape[0], mat.shape[1]))
    return compactness, labels, centers


def mask_from_labels(labels: np.ndarray,
                     centers: np.ndarray) -> List[np.ndarray]:
    """Generates masks from labels generated from k-means.

    Args:
        labels: The labels generated by k-means.
        centers: The centeroids generated by k-means.

    Returns:
        A list of masks, one for each centeroid in centers. For each mask, 
        all points in the same label are set to 255, while all other points are
        set to 0.
    """
    acc = []
    for i, c in enumerate(centers):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels==i] = 255
        acc.append(mask)
    return acc


def mask_from_labels_target_color(labels: np.ndarray,
                                  centers: np.ndarray,
                                  target_color: Tuple[int, int, int], 
                                  distance_func: Callable = color_dist) -> np.ndarray:
    """Generates a mask from labels generated from k-means based on the closest
    label to a target color.

    Args:
        labels: The labels generated by k-means.
        centers: The centeroids generated by k-means.
        target_color: The target color used to choose the label generating the masks.
        distance_func: The function that takes in two 3-channel colors to calculate 
                       the color distance of the colors.

    Returns:
        A mask where all points in the label closest to the target color are set
        to 255, while all other points are set to 0.
    """
    target_label = min(enumerate(centers), key= lambda x: distance_func(x, target_color))
    ret = np.zeros(labels.shape, dtype=np.uint8)
    ret[labels==target_label] = 255
    return ret

def white_balance_bgr(bgr_img):
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_l, lab_a, lab_b = cv2.split(lab_img)
    a_avg = np.mean(lab_a)
    b_avg = np.mean(lab_b)
    lab_a -= a_avg - 128
    lab_b -= b_avg - 128
    lab_img = cv2.merge((lab_l, lab_a, lab_b))
    return cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)


def white_balance_bgr_blur(bgr_img, kernel_size):
    kernel_size //= 2
    kernel_size = 2 * kernel_size + 1
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_l, lab_a, lab_b = cv2.split(lab_img)
    lab_a_avg = cv2.blur(lab_a, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_REPLICATE)
    lab_b_avg = cv2.blur(lab_b, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_REPLICATE)
    lab_a -= lab_a_avg - 128
    lab_b -= lab_b_avg - 128
    lab_img = cv2.merge((lab_l, lab_a, lab_b))
    return cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)

