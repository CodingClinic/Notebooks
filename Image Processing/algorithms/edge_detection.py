import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def detect_edges(path_to_image: str, show_chart: bool = True):
    """
    Canny Edge Detection

    References:
    * https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    """
    img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(img, 100, 200)

    if show_chart:
        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap="gray")
        plt.title("Edge Image"), plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == "__main__":
    abnormal_mri_image = "../images/Abnormal MRI Raw.png"
    healthy_mri_image = "../images/Healthy MRI Raw.png"

    # detect_edges(abnormal_mri_image)
    detect_edges(healthy_mri_image)
