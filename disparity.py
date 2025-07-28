import cv2
import numpy as np
import time 
import matplotlib.pyplot as plt

def generate_window(row, col, image, blockSize):
    window = (image[row:row + blockSize, col:col + blockSize])
    return window

def wls_disparity(limg, rimg):

    lmbda = 8000
    sigma = 1.5

    #create stereo matcher
    stereo_left = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*9,  # Must be a multiple of 16
    blockSize=11,
    P1=8*11*11,
    P2=32*11*11,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

    # Compute disparity maps
    disparity_left = stereo_left.compute(limg, rimg).astype(np.float32) / 16.0
    disparity_right = stereo_right.compute(rimg, limg).astype(np.float32) / 16.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    wls_filter.setLambda(lmbda)  # Smoothness strength
    wls_filter.setSigmaColor(sigma)  # Edge-preserving filter strength

    wls_disparity = wls_filter.filter(disparity_left, limg, disparity_map_right=disparity_right)

    confidence = wls_filter.getConfidenceMap()
    wls_disparity[confidence<10] = float('nan')

    # plt.imshow(wls_disparity)
    # plt.show()

    return wls_disparity

def subpixel_disparity(limg, rimg):

    #create stereo matcher
    stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,  # Must be a multiple of 16
    blockSize=9,
    P1=9 * 3 * 2,
    P2=64 * 3 * 2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63)

    # Compute disparity (fixed-point format, multiply by 16 for precision)
    disparity = stereo.compute(limg, rimg).astype(np.float32) / 16.0

    # Apply bilateral filter to smooth disparities
    subpixel_disparity = cv2.bilateralFilter(disparity, d=15, sigmaColor=25, sigmaSpace=25)

    # plt.imshow(subpixel_disparity)
    # plt.show()

    return subpixel_disparity