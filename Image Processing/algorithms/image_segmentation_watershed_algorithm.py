import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def run_watershed_segmentation_algorithm(path_to_image: str, show_chart: bool = True):
    """
    References:
    * https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    * https://people.cmm.minesparis.psl.eu/users/beucher/wtshed.html
    """
    img = cv.imread(path_to_image)
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    if show_chart:
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    abnormal_mri_image = "../images/Abnormal MRI Raw.png"
    healthy_mri_image = "../images/Healthy MRI Raw.png"

    run_watershed_segmentation_algorithm(abnormal_mri_image)
    # run_watershed_segmentation_algorithm(healthy_mri_image)
