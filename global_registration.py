import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 3
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_cloud(voxel_size, i):
    
    print(":: Load  point cloud and disturb initial pose.")
    cloud = o3d.io.read_point_cloud(f"mala_cloud/m{i}.ply")

    R = np.identity(3)  
    extent = np.ones(3)*0.65 # trying to create a bounding box below 1 unit
    center = np.ones(3) 
    obb = o3d.geometry.OrientedBoundingBox([0,0,-0.5],R,extent) # or you can use axis aligned bounding box class

    #cloud = pcd.crop(obb)


    #draw_registration_result(source, target, np.identity(4))

    target_down, target_fpfh = preprocess_point_cloud(cloud, voxel_size)
    return cloud, target_down, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 4
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, True,  # Mutual filter should be a boolean
    distance_threshold,  
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4,
    [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    o3d.pipelines.registration.RANSACConvergenceCriteria(600000, 800))
    return result
