import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def translate(points, translation):
    return points + translation


def rotate(points, rotation_matrix):
    return (rotation_matrix @ points.T).T


def view_points(points, intrinsic_matrix):
    points = rotate(points, intrinsic_matrix)
    return points / points[:, 2:3]


def dummy_segmentation_logits(image_shape, num_classes=10):
    H, W = image_shape[:2]
    return np.random.rand(H, W, num_classes)


DATA_ROOT = "../data/sets/nuscenes"
nusc = NuScenes(version="v1.0-mini", dataroot=DATA_ROOT, verbose=True)

my_scene = nusc.scene[0]
first_sample_token = my_scene["first_sample_token"]
my_sample = nusc.get("sample", first_sample_token)

lidar_sensor = "LIDAR_TOP"
lidar_data = nusc.get("sample_data", my_sample["data"][lidar_sensor])

lidar_path = os.path.join(DATA_ROOT, lidar_data["filename"])
bin_pcd = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))[:, :3]

# get lidar calibration data & ego pose at the lidar timestamp
cs_lidar = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
pose_lidar = nusc.get("ego_pose", lidar_data["ego_pose_token"])

# transform lidar to global frame
bin_pcd_global = rotate(bin_pcd, Quaternion(cs_lidar["rotation"]).rotation_matrix)
bin_pcd_global = translate(bin_pcd_global, np.array(cs_lidar["translation"]))
bin_pcd_global = rotate(
    bin_pcd_global, Quaternion(pose_lidar["rotation"]).rotation_matrix
)
bin_pcd_global = translate(bin_pcd_global, np.array(pose_lidar["translation"]))

painted_points_all = []
num_classes = 10

camera_sensors = [key for key in my_sample["data"] if key.startswith("CAM_")]

for cam_sensor in camera_sensors:
    cam_data = nusc.get("sample_data", my_sample["data"][cam_sensor])
    cs_cam = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    pose_cam = nusc.get("ego_pose", cam_data["ego_pose_token"])

    # global to ego at the camera timestamp
    points_ego_cam = translate(bin_pcd_global, -np.array(pose_cam["translation"]))
    points_ego_cam = rotate(
        points_ego_cam, Quaternion(pose_cam["rotation"]).rotation_matrix.T
    )

    # ego to camera frame
    points_cam_frame = translate(points_ego_cam, -np.array(cs_cam["translation"]))
    points_cam_frame = rotate(
        points_cam_frame, Quaternion(cs_cam["rotation"]).rotation_matrix.T
    )

    # Load image and dummy logits
    img_path = os.path.join(DATA_ROOT, cam_data["filename"])
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    logits_map = dummy_segmentation_logits(img.shape, num_classes=num_classes)

    # project to 2d
    projected_pts = view_points(points_cam_frame, np.array(cs_cam["camera_intrinsic"]))

    # mask: take whats in front of the camera & inside image bounds
    valid_mask = (
        (points_cam_frame[:, 2] > 0.5)
        & (projected_pts[:, 0] >= 0)
        & (projected_pts[:, 0] < img_width)
        & (projected_pts[:, 1] >= 0)
        & (projected_pts[:, 1] < img_height)
    )

    # apply mask to projected points
    projected_int = projected_pts[valid_mask].astype(int)

    # retrieve logits for valid pixels
    point_logits = logits_map[projected_int[:, 1], projected_int[:, 0], :]

    # apply mask to points in the camera frame
    valid_points_cam = points_cam_frame[valid_mask]

    # camera frame to ego at the camera timestamp
    valid_points_ego = rotate(
        valid_points_cam, Quaternion(cs_cam["rotation"]).rotation_matrix
    )
    valid_points_ego = translate(valid_points_ego, np.array(cs_cam["translation"]))

    # ego at the camera timestamp to global frame
    valid_points_global = rotate(
        valid_points_ego, Quaternion(pose_cam["rotation"]).rotation_matrix
    )
    valid_points_global = translate(
        valid_points_global, np.array(pose_cam["translation"])
    )

    # global frame to ego at the lidar timestamp
    valid_points_ego_lidar = translate(
        valid_points_global, -np.array(pose_lidar["translation"])
    )
    valid_points_ego_lidar = rotate(
        valid_points_ego_lidar, Quaternion(pose_lidar["rotation"]).rotation_matrix.T
    )

    # ego to lidar frame
    valid_points_lidar = translate(
        valid_points_ego_lidar, -np.array(cs_lidar["translation"])
    )
    valid_points_lidar = rotate(
        valid_points_ego_lidar, Quaternion(cs_lidar["rotation"]).rotation_matrix.T
    )

    painted_points = np.concatenate([valid_points_lidar, point_logits], axis=1)

    painted_points_all.append(painted_points)

if painted_points_all:
    painted_points_all = np.vstack(painted_points_all)
