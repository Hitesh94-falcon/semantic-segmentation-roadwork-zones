#!/usr/bin/env python3

"""
    @author: Numan Senel
    @email: Numan.Senel@thi.de
"""

import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2 as pc2

def points_on_image(undistorted_img, point_cloud, rotation_translation, intrinsic, veloyne=False):
    color_scale = 255 / 3
    p = np.matmul(intrinsic, rotation_translation)
    test_array = np.array(list(pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
    
    test_array = np.transpose(test_array)
    reflection = test_array[3, :].copy()
    test_array[3, :] = 1
    test_array = np.matmul(p, test_array)
    
    test_array = np.array([test_array[0, :] / test_array[2, :],
                           test_array[1, :] / test_array[2, :],
                           reflection]).T
    test_array = test_array.astype(int)
    
    (rows, cols, channels) = undistorted_img.shape
    for cor in test_array:
        px_c, px_r, color = cor
        
        # Ensure the color value is an integer and clamp it within a valid range (0 to 255)
        color = int(np.clip(color, 0, 255))
        
        if 0 <= px_c < cols and 0 <= px_r < rows:
            cv2.circle(undistorted_img, (px_c, px_r), 1, (140, 70, color), -1)
    
    cv2.imshow("Lidar points on image", undistorted_img)
    cv2.waitKey(3)

# import numpy as np
# import cv2

# # Try to import ROS PointCloud2 tools (won’t break if ROS not installed)
# try:
#     from sensor_msgs_py import point_cloud2 as pc2
#     from sensor_msgs.msg import PointCloud2
#     ROS_AVAILABLE = True
# except ImportError:
#     ROS_AVAILABLE = False


# def points_on_image(undistorted_img, point_cloud, rotation_translation, intrinsic):
#     """
#     Project LiDAR points onto an undistorted camera image.

#     Args:
#         undistorted_img (np.ndarray): BGR image already undistorted.
#         point_cloud (sensor_msgs.msg.PointCloud2 | np.ndarray): 
#             Either a ROS PointCloud2 message or Nx4 numpy array [x, y, z, intensity].
#         rotation_translation (np.ndarray): 3x4 LiDAR→Camera extrinsic [R|t].
#         intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
#     """

#     # --- 1. Convert point cloud to numpy (N,4) ---
#     if ROS_AVAILABLE and isinstance(point_cloud, PointCloud2):
#         # Read ROS PointCloud2 message
#         data = list(pc2.read_points(
#             point_cloud,
#             field_names=("x", "y", "z", "intensity"),
#             skip_nans=True
#         ))
#         # Convert to regular float32 Nx4 array
#         pts = np.stack([np.array([p[0], p[1], p[2], p[3]], dtype=np.float32) for p in data])

#     elif isinstance(point_cloud, np.ndarray):
#             pts = point_cloud.astype(np.float32)
#             if pts.shape[1] < 4:
#                 # add fake intensity if not provided
#                 intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
#                 pts = np.hstack((pts, intensity))
#     else:
#         raise TypeError("point_cloud must be PointCloud2 or Nx4 numpy array")

#     # --- 2. Extract and filter points ---
#     points3d = pts[:, :3].T  # shape (3, N)
#     intensity = pts[:, 3]

#     # Only use points in front of the camera
#     mask = points3d[2, :] > 0
#     points3d = points3d[:, mask]
#     intensity = intensity[mask]

#     # --- 3. Project LiDAR → Camera → Image plane ---
#     P = intrinsic @ rotation_translation  # 3x4 projection
#     proj = P @ np.vstack((points3d, np.ones((1, points3d.shape[1]))))  # 3xN

#     # Normalize
#     proj[:2, :] /= proj[2, :]

#     # --- 4. Keep points within image bounds ---
#     h, w = undistorted_img.shape[:2]
#     u, v = proj[0, :].astype(np.int32), proj[1, :].astype(np.int32)
#     valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
#     u, v, intensity = u[valid], v[valid], intensity[valid]

#     # --- 5. Color mapping by intensity ---
#     intensity_norm = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     colors = cv2.applyColorMap(intensity_norm, cv2.COLORMAP_JET)

#     # --- 6. Draw points on image ---
#     img_out = undistorted_img.copy()
#     for i in range(len(u)):
#         color = tuple(int(c) for c in colors[i, 0])
#         cv2.circle(img_out, (u[i], v[i]), 2, color, -1)

#     return img_out



