#!/usr/bin/env python3

#author: Zhiran Yan
#Email: zhiran.yan@thi.de

# three paths: 1. intrinsic 2. extrinsic result 3. clicked points

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import namedtuple
import yaml , json


# Define simple 2D and 3D point containers
Point2D = namedtuple('Point2D', 'x y')
Point3D = namedtuple('Point3D', 'x y z')
np.set_printoptions(suppress=True)


with open("/home/hitesh/Documents/Project/multi_target_calibration/filepaths.json", "r") as file:
    paths = json.load(file)

global calibrate
calibrate = True


class BoundingBoxFilterNode(Node):
    def __init__(self):
        super().__init__('image_point_lidar_calibrator')
        self.window_name = "Test"
        self.calibrated_number = 0
        self.bridge = CvBridge()
        

        # Create subscribers and publisher with appropriate QoS profiles
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(
            Image,
            paths["ros2_nodes"]["camera_node"],
            self.image_callback,
            qos_profile_sensor_data
        )
        self.image_pub = self.create_publisher(Image, '/image_out', 10)
        self.create_subscription(
            PointStamped,
            paths["ros2_nodes"]['clicked_point'],
            self.clicked_point_callback,
            10  # QoS for non-sensor data
        )
        self.create_subscription(
            pc2,
            paths["ros2_nodes"]["lidar_points_node"],
            self.lidar_callback,
            qos_profile_sensor_data
        )

        # Load intrinsic parameters from the JSON file
        # self.load_intrinsic(
        #     "/media/zhiranworkstation/T7a/group_project/multi_target_calibration/cam1/intrinsic_cam1.json"
        # )
        
        self.load_intrinsicyaml(paths["calibration_file_paths"]["intrinsic_file_path"])

        # Initialize image, point cloud, and point storage
        self.image = np.zeros((500, 500, 3), dtype=np.uint8)
        self.point_cloud = None
        self.points2d = []
        self.points3d = []

        # Create a timer for periodic image display and calibration check (10 Hz)
        self.create_timer(0.1, self.show_image)

    # def load_intrinsic(self, json_file_path):
    #     with open(json_file_path, 'r') as file:
    #         data = json.load(file)
    #         intrinsic_data = data["center_camera-intrinsic"]["param"]["cam_K"]["data"]
    #         self.intrinsic = np.array(intrinsic_data, dtype=float).reshape(3, 3)
    #         distCoeffs_data = data["center_camera-intrinsic"]["param"]["cam_dist"]["data"][0]
    #         self.distCoeffs = np.array([distCoeffs_data], dtype=float)
    #     self.get_logger().info("Loaded intrinsic parameters.")

    def load_intrinsicyaml(self,yaml_file_path):
        try:
            with open(yaml_file_path, "r") as file:
                data = yaml.safe_load(file)
                self.intrinsic = np.array(data['camera_matrix']['data']).reshape(3, 3)
                self.distCoeffs = np.array(data['distortion_coefficients']['data'], dtype=float)
            self.get_logger().info("Loaded intrinsic parameters from YAML.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML file: {e}")

    def save_rotation_to_json(self, rotation_matrix, file_path):
        if rotation_matrix is None:
            self.get_logger().error("Received 'None' for rotation_matrix. Aborting save to JSON.")
            return
        json_dict = {
            "top_center_lidar-to-cam1-extrinsic": {
                "sensor_name": "top_center_lidar",
                "target_sensor_name": "camera7",
                "device_type": "relational",
                "param_type": "extrinsic",
                "param": {
                    "time_lag": 0,
                    "sensor_calib": {
                        "rows": 4,
                        "cols": 4,
                        "type": 6,
                        "continuous": True,
                        "data": rotation_matrix.tolist()
                    }
                }
            }
        }
        with open(file_path, 'w') as file:
            json.dump(json_dict, file, indent=4)
        self.get_logger().info(f"Extrinsic matrix saved to {file_path}")

    def save_clicked_points(self, file_path):
        clicked_points = {
            "clicked_points": {
                "3D_points": [{"x": pt.x, "y": pt.y, "z": pt.z} for pt in self.points3d],
                "2D_points": [{"u": pt.x, "v": pt.y} for pt in self.points2d]
            }
        }
        with open(file_path, 'w') as file:
            json.dump(clicked_points, file, indent=4)
        self.get_logger().info(f"Clicked points saved to {file_path}")

    def lidar_callback(self, cloud_msg):
        self.point_cloud = cloud_msg
        self.get_logger().debug("Received point cloud.")

    def clicked_point_callback(self, point_msg):
        self.get_logger().info(f"Received clicked point: {point_msg}")
        self.points3d.append(Point3D(point_msg.point.x, point_msg.point.y, point_msg.point.z))
        self.get_logger().info(f"Number of 2D points: {len(self.points2d)}; Number of 3D points: {len(self.points3d)}")

    def cloudtoImage(self, pointcloud, transformation_matrix, intrinsic):
        # If transformation_matrix is 4x4, reduce it to 3x4 for projection.
        if transformation_matrix.shape == (4, 4):
            transformation_matrix = transformation_matrix[:3, :]
        p = np.matmul(intrinsic, transformation_matrix)
        self.get_logger().debug(f"Projection matrix p: {p}")
        # Add homogeneous coordinate if necessary.
        if pointcloud.shape[0] == 3:
            pointcloud = np.vstack([pointcloud, np.ones((1, pointcloud.shape[1]))])
        test_array = np.matmul(p, pointcloud)
        test_array /= test_array[2, :]
        return test_array[:2, :].astype(int)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.image, (x, y), 10, (0, 70, 70), -1)
            self.points2d.append(Point2D(x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.get_logger().info(f"2D points: {len(self.points2d)}; 3D points: {len(self.points3d)}")
        self.get_logger().debug(f"Current 2D points: {self.points2d}")
        self.get_logger().debug(f"Current 3D points: {self.points3d}")

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        # Undistort the image using the loaded intrinsic parameters.
        cv_image = cv2.undistort(cv_image, self.intrinsic, self.distCoeffs)
        self.get_logger().debug(f"Received image shape: {cv_image.shape}")
        self.image = cv_image.copy()
        # Redraw any previously clicked 2D points.
        for point in self.points2d:
            cv2.circle(self.image, (point.x, point.y), 3, (0, 70, 70), -1)

    def show_image(self):
        rescaled_image = cv2.resize(self.image,(1920,1200))
        cv2.imshow(self.window_name, rescaled_image)
        cv2.setMouseCallback(self.window_name, self.mouse_event)
        cv2.waitKey(1)

        if calibrate:
            # Check if enough corresponding points have been collected to perform calibration.
            if (len(self.points3d) > 9 and 
                len(self.points3d) == len(self.points2d) and 
                self.calibrated_number != len(self.points3d)):
                transformation_matrix = self.calibrate()
                if transformation_matrix is not None:
                    self.calibrated_number = len(self.points3d)
                    self.save_rotation_to_json(
                        transformation_matrix,
                        paths["calibration_file_paths"]["extrinic_calibrated_path"]
                    )
                    # Visualize lidar points on the current image immediately after calibration
                    if self.point_cloud is not None:
                        self.points_on_image(self.image.copy(), self.point_cloud, transformation_matrix, self.intrinsic)

            # If we already have a computed transform, keep projecting incoming clouds each loop
            elif hasattr(self, "rotation_translation") and self.point_cloud is not None:
                self.points_on_image(self.image.copy(), self.point_cloud, self.rotation_translation, self.intrinsic)

    def calibrate(self):
        if calibrate:
            self.get_logger().info(f"Calibrating with {len(self.points3d)} 3D points and {len(self.points2d)} 2D points.")
            world_points = []
            image_points = []
            for idx in range(len(self.points3d)):
                pt3d = self.points3d[idx]
                pt2d = self.points2d[idx]
                world_points.append([pt3d.x, pt3d.y, pt3d.z])
                image_points.append([pt2d.x, pt2d.y])
            objectPoints = np.array(world_points, dtype=float).reshape(len(self.points3d), 3, 1)
            imagePoints = np.array(image_points, dtype=float).reshape(len(self.points2d), 2, 1)
            # Use zero distortion for calibration since we already undistort the images.
            distCoeffs_zero = np.zeros((5, 1))
            # Perform PnP RANSAC calibration using the P3P method.
            success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                objectPoints,
                imagePoints,
                self.intrinsic,
                distCoeffs_zero,
                flags=cv2.SOLVEPNP_P3P,
                iterationsCount=1000
            )
            if success:
                R_mat, _ = cv2.Rodrigues(R_vec)
                self.rotation_translation = np.hstack((R_mat, t_vec.reshape(-1, 1)))
                transformation_matrix = np.vstack((self.rotation_translation, [0, 0, 0, 1]))
                
                # Project the collected 3D points onto the image for verification.
                np_world_points = np.array([
                    [pt.x for pt in self.points3d],
                    [pt.y for pt in self.points3d],
                    [pt.z for pt in self.points3d],
                    [1 for _ in self.points3d]
                ], dtype=float)
                projected_2d_points = self.cloudtoImage(np_world_points, transformation_matrix, self.intrinsic)
                self.get_logger().info(f"Final projected 2D points on the image plane: {projected_2d_points}")
                clicked_2d_points = np.array([[pt.x for pt in self.points2d], [pt.y for pt in self.points2d]])
                self.get_logger().info(f"Difference (projected - clicked): {projected_2d_points - clicked_2d_points}")
                return transformation_matrix
            else:
                self.get_logger().error("Calibration failed.")
                return self.rotation_translation
    def points_on_image(self, undistorted_img, point_cloud, rotation_translation, intrinsic, veloyne=False):
        # Ensure numpy arrays
        intrinsic = np.asarray(intrinsic, dtype=float)
        rt = np.asarray(rotation_translation, dtype=float)

        # Accept 4x4 homogeneous, 3x4 extrinsic, or 3x3 rotation (append zero translation)
        if rt.shape == (4, 4):
            rt3x4 = rt[:3, :]
        elif rt.shape == (3, 4):
            rt3x4 = rt
        elif rt.shape == (3, 3):
            rt3x4 = np.hstack((rt, np.zeros((3, 1), dtype=float)))
        else:
            raise ValueError(f"Unsupported rotation_translation shape {rt.shape}; expected (4,4) or (3,4) or (3,3)")

        # Projection matrix 3x4
        P = intrinsic @ rt3x4

        # Read point cloud into NxM list safely
        pts = list(pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z", "intensity")))
        if len(pts) == 0:
            return
        arr = np.array(pts).T  # shape could be (3,N) or (4,N)

        # Ensure homogeneous 4xN (x,y,z,1) and keep intensity if present
        if arr.shape[0] == 3:
            intens = np.zeros(arr.shape[1], dtype=float)
            hom = np.vstack((arr, np.ones((1, arr.shape[1]))))
        else:
            intens = arr[3, :].copy()
            hom = arr.copy()
            hom[3, :] = 1.0

        # Project: 3x4 @ 4xN -> 3xN
        proj = P @ hom
        z = proj[2, :]
        valid = z != 0
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[valid] = (proj[0, valid] / z[valid])
        v[valid] = (proj[1, valid] / z[valid])

        # prepare int coords and colors
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        colors = intens.astype(int) if intens is not None else np.zeros_like(u)

        h, w = undistorted_img.shape[:2]
        for uu, vv, col, ok in zip(u, v, colors, valid):
            if not ok:
                continue
            if 0 <= uu < w and 0 <= vv < h:
                c = int(np.clip(col, 0, 255))
                cv2.circle(undistorted_img, (uu, vv), 1, (140, 70, c), -1)

        cv2.imshow("Lidar points on image", undistorted_img)
        cv2.waitKey(1)


def main(args=None):

        rclpy.init(args=args)
        node = BoundingBoxFilterNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard Interrupt, shutting down.")
        finally:
            # Save the clicked points before shutting down.
            node.save_clicked_points(
                paths["calibration_file_paths"]["clicked_points"]
            )
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()



if __name__ == '__main__':
    main()