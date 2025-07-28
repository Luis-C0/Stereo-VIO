import cv2
import numpy as np


def rectify_stereo_images(left_img, right_img, mapx1, mapy1, mapx2, mapy2):
    """
    Rectify stereo image pair
    """
    rect_left = cv2.remap(left_img, mapx1, mapy1, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, mapx2, mapy2, cv2.INTER_LINEAR)
    
    return rect_left, rect_right


