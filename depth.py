from math import nan
import numpy as np
import cv2
import matplotlib.pyplot as plt


baseline = 50

def depth_map(dispMap, focal_lenght):
    # print("Calculating depth....")
    
    depth = (focal_lenght * baseline) / dispMap

    depth = np.clip(depth, 600, 1500)
    # depthmap = plt.imshow(depth,cmap='jet_r')
    # plt.colorbar(depthmap)
    # plt.show()
    
    return depth[50:900,650:1450]

