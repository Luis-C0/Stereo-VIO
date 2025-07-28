import os
import ast
import csv
from typing import final
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy import signal
from math import pi
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Mahony
from tqdm import tqdm
from calibrate import rectify_stereo_images
from disparity import wls_disparity
from depth import depth_map
from cloud_generation import get_points

# filepath = "test-Depth"
filepath = "test_horiz"
#filepath = "test_vert"
#filepath = "test_prox"

def convert_cloud(trans):
    T = np.eye(4)
    T[2,:] = trans[0,:]
    T[1,:] = trans[2,:]
    T[0,:] = trans[1,:]
    return T

def convert_imu(trans):
    T = np.eye(4)
    T[0,:] = trans[2,:]
    T[1,:] = trans[0,:]
    T[2,:] = trans[1,:]
    return T

# --- Load IMU calibration data ---
with open('imu_offsets.txt', 'r') as file:
    lines = file.readlines()

gyrxdisp = float(lines[0].strip())
gyrydisp = float(lines[1].strip())
gyrzdisp = float(lines[2].strip())
accxcor = ast.literal_eval(lines[3].strip())
accycor = ast.literal_eval(lines[4].strip())
acczcor = ast.literal_eval(lines[5].strip())

# --- Load IMU CSV ---
file = f'{filepath}/imu_data.csv'
startTime = 3
samplePeriod = 1 / 140

data = np.loadtxt(file, delimiter=',', skiprows=1)
timestamp = data[:, 0]
gyrx = (data[:, 4] - gyrxdisp) * np.pi / 180
gyry = (data[:, 5] - gyrydisp) * np.pi / 180
gyrz = (data[:, 6] - gyrzdisp) * np.pi / 180
accx = data[:, 1] * accxcor[0] + accxcor[1]
accy = data[:, 2] * accycor[0] + accycor[1]
accz = data[:, 3] * acczcor[0] + acczcor[1]

# --- Filter accelerometer magnitude ---
acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)

# High-pass filter
filtCutOff = 0.001
b, a = signal.butter(1, (2 * filtCutOff) / (1 / samplePeriod), 'highpass')
acc_magFilt = signal.filtfilt(b, a, acc_mag, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
acc_magFilt = np.abs(acc_magFilt)

# Low-pass filter
filtCutOff = 5
b, a = signal.butter(1, (2 * filtCutOff) / (1 / samplePeriod), 'lowpass')
acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))

# --- Initial orientation estimate ---
indexInit = timestamp <= startTime
gyr = np.array([
    np.mean(gyrx[indexInit]) / 100,
    np.mean(gyry[indexInit]) / 100,
    np.mean(gyrz[indexInit]) / 100
])
acc = np.array([
    np.mean(accx[indexInit]),
    np.mean(accy[indexInit]),
    np.mean(accz[indexInit])
])

orientation = Mahony(frequency=140.0)
q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

for _ in range(2000):
    q = orientation.updateIMU(q=q, gyr=gyr, acc=acc)

Tin = np.eye(4)
Tin[:3,:3] = R.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()

# --- IMU Integrator ---
class IMUIntegrator:
    def __init__(self):
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.gravity = np.array([0, 0, -9.9])
        self.global_pos = Tin.copy()
        self.ori_filter = Mahony(frequency=140.0)

    def integrate(self, imu_data):
        times = imu_data[:, 0]
        gyro_data = imu_data[:, 1:4]
        accel_data = imu_data[:, 4:7]
        pos = np.zeros(3)
        quat = R.from_matrix(self.global_pos[:3, :3]).as_quat()
        q = np.array([quat[3],quat[0],quat[1],quat[2]])

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            q = self.ori_filter.updateIMU(q, gyr=gyro_data[i], acc=accel_data[i])
            Rwb = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            gravity_body = Rwb.T @ self.gravity
            acc_corrected = accel_data[i] - gravity_body

            self.velocity += acc_corrected * dt
            pos += self.velocity * dt + 0.5 * acc_corrected * dt**2

        T_relative = np.eye(4)
        self.global_pos[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T_relative[:3, 3] = pos
        self.global_pos = self.global_pos @ T_relative

        return self.global_pos

    def update(self,transform):
        self.position = transform[:3,3].copy()
        self.global_pos = transform.copy()

imu_tracker = IMUIntegrator()

# --- Load rectification maps ---
fs = cv2.FileStorage("rectification_maps.yaml", cv2.FILE_STORAGE_READ)
K = fs.getNode("K").mat()
mapx1 = fs.getNode("mapx1").mat()
mapx2 = fs.getNode("mapx2").mat()
mapy1 = fs.getNode("mapy1").mat()
mapy2 = fs.getNode("mapy2").mat()
fs.release()

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]


# --- Load image filenames ---
lefts = sorted([f for f in os.listdir(f"{filepath}/images_left") if f.endswith(".jpg")], key=lambda x: float(os.path.splitext(x)[0]))
rights = sorted([f for f in os.listdir(f"{filepath}/images_right") if f.endswith(".jpg")], key=lambda x: float(os.path.splitext(x)[0]))

# --- Initial frame setup ---
j = 10
threshold = 5  # mm

imgl = cv2.imread(os.path.join(f"{filepath}/images_left", lefts[j]))
imgr = cv2.imread(os.path.join(f"{filepath}/images_right", rights[j]))
time1 = float(os.path.splitext(rights[j])[0])

gray_left = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
rect_left, rect_right = rectify_stereo_images(gray_left, gray_right, mapx1, mapy1, mapx2, mapy2)

Disp_Image = wls_disparity(rect_left, rect_right)
Depth_Map = depth_map(Disp_Image,fx)
target_pcd = get_points(Depth_Map,fx,fy,cx,cy)
target_pcd.transform(convert_cloud(Tin))
target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30))
final_pcd = target_pcd
print(imu_tracker.position)

# --- Iterate through frames ---
for i in tqdm(range(j + 1, len(lefts),10)):
    imgl = cv2.imread(os.path.join(f"{filepath}/images_left", lefts[i]))
    imgr = cv2.imread(os.path.join(f"{filepath}/images_right", rights[i]))
    time2 = float(os.path.splitext(rights[i])[0])

    gray_left = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    rect_left, rect_right = rectify_stereo_images(gray_left, gray_right, mapx1, mapy1, mapx2, mapy2)

    Disp_Image = wls_disparity(rect_left, rect_right)
    Depth_Map = depth_map(Disp_Image,fx)
    source_pcd = get_points(Depth_Map,fx,fy,cx,cy)
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30))

    # Get IMU data for current interval
    indexSel = (timestamp >= time1) & (timestamp <= time2)
    imu_data = np.stack([
        timestamp[indexSel],
        gyrx[indexSel], gyry[indexSel], gyrz[indexSel],
        accx[indexSel], accy[indexSel], accz[indexSel]
    ], axis=-1)

    old_pos = imu_tracker.position

    # Align clouds with IMU-predicted pose
    initial_transform = convert_cloud(imu_tracker.integrate(imu_data))
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, final_pcd, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=300
        )
    )

    source_pcd.transform(reg_p2p.transformation)
    target_pcd = source_pcd
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30))
    final_pcd += target_pcd
    print("Fitness:", reg_p2p.fitness)

    # Update velocity from displacement
    imu_tracker.update(convert_imu(reg_p2p.transformation))
    imu_tracker.velocity = 0

    time1 = time2

o3d.visualization.draw_geometries([final_pcd])
print(imu_tracker.position)
o3d.io.write_point_cloud("Dist.ply", final_pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)