import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from calibrate import *
from disparity import disparitymap, left_right_disparity, wls_disparity, subpixel_disparity
from depth import depth_map
from cloud_generation import get_points
from tqdm import tqdm
from global_registration import *

j = 37
voxel_size = 10  # means 5mm for the dataset
threshold = 5

target = o3d.io.read_point_cloud(f"teste2/wls10/{j}.ply")
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20,max_nn=30))
target_down, target_fpfh = preprocess_point_cloud(target,voxel_size)
pcd = target

for i in tqdm(range(j-1,j-20,-1)):

    source = o3d.io.read_point_cloud(f"teste2/wls10/{i}.ply")

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20,max_nn=30))    
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    trans = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold,trans.transformation,o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, max_iteration=1000))
    target = source.transform(reg_p2p.transformation)
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20,max_nn=30))
    target_down, target_fpfh = preprocess_point_cloud(target,voxel_size)
    pcd += target
    print("Fitness:", reg_p2p.fitness)

pcdown, pcdfpfh = preprocess_point_cloud(pcd, voxel_size*0.25)
o3d.visualization.draw_geometries([pcd])
# o3d.io.write_point_cloud("3dfinal.ply", pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)

