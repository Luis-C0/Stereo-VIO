import numpy as np
import open3d as o3d


def get_points(DMap,fx,fy,cx,cy):
    
    height,width = DMap.shape[0], DMap.shape[1]
    points = []

    for row in range(height):
        for col in range(width):

            X = (col - cx) * DMap[row,col] / fx
            Y = (row - cy) * DMap[row,col] / fy
            Z = DMap[row,col]
            if Z<1300:
                points.append([X,Y,Z])
    
    points = np.array(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=0.5)
    # o3d.visualization.draw_geometries([cl])
    return cl
