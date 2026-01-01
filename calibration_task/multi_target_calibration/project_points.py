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
import math


# Define simple 2D and 3D point containers
Point2D = namedtuple('Point2D', 'x y')
Point3D = namedtuple('Point3D', 'x y z')
np.set_printoptions(suppress=True)


with open("/home/hitesh/Documents/Project/multi_target_calibration/filepaths.json", "r") as file:
    paths = json.load(file)



class Points_on_image(Node):
    def __init__(self):
        super().__init__('image_point_lidar_')
        self.window_name = "lidar_to_image"
        self.bridge = CvBridge()
        

        # Create subscribers and publisher with appropriate QoS profiles
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(
            Image,
            paths["ros2_nodes"]["camera_node"],
            self.image_callback,
            qos_profile_sensor_data
        )

        # subscribe using the ROS2 message class PointCloud2 (pc2 is the helper module)
        self.create_subscription(
            PointCloud2,
            paths["ros2_nodes"]["lidar_points_node"],
            self.lidar_callback,
            qos_profile_sensor_data
        )

        self.image = np.zeros((500, 500, 3), dtype=np.uint8)

        self.create_timer(0.1, self.show_image)


        self.load_intrinsicyaml(paths["calibration_file_paths"]["intrinsic_file_path"])

    def load_intrinsicyaml(self,yaml_file_path):
        try:
            with open(yaml_file_path, "r") as file:
                data = yaml.safe_load(file)
                self.intrinsic = np.array(data['camera_matrix']['data']).reshape(3, 3)
                self.distCoeffs = np.array(data['distortion_coefficients']['data'], dtype=float)
            self.get_logger().info("Loaded intrinsic parameters from YAML.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML file: {e}")
        

    def lidar_callback(self, cloud_msg):
        self.point_cloud = cloud_msg
        self.get_logger().debug("Received point cloud.")

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        cv_image = cv2.undistort(cv_image, self.intrinsic, self.distCoeffs)
        self.get_logger().debug(f"Received image shape: {cv_image.shape}")
        self.image = cv_image.copy()

    def rotation_translation(self):
        self.fp = paths["calibration_file_paths"]["calibrated_json"]
        try:
            with open(self.fp,"r") as f:
                doc = json.load(f)
            node = doc["calculated_error_matrix_camera_to_lidar"]
            R = np.array(node["rotation"], dtype=float)    
            t = np.array(node["translation"], dtype=float) 
            self.r_t = np.hstack((R, t.reshape(3, 1)))     
            return self.r_t
        except Exception as e:
            self.get_logger().error(f"failed to read the json from {self.fp}: {e}")

    def show_image(self):
        if not hasattr(self, "r_t"):
            try:
                self.rotation_translation()
            except Exception as e:
                self.get_logger().debug(f"No transform yet: {e}")
                return
        if hasattr(self, "point_cloud") and self.point_cloud is not None:
            self.points_on_image(self.image, self.point_cloud, self.r_t, self.intrinsic)


    def points_on_image(self, undistorted_img, point_cloud, rotation_translation, intrinsic, veloyne=False):
   
        intrinsic = np.asarray(intrinsic, dtype=float)
        rt = np.asarray(rotation_translation, dtype=float)

        if rt.shape == (4, 4):
            rt3x4 = rt[:3, :]
        elif rt.shape == (3, 4):
            rt3x4 = rt
        elif rt.shape == (3, 3):
            rt3x4 = np.hstack((rt, np.zeros((3, 1), dtype=float)))
        else:
            raise ValueError(f"Unsupported rotation_translation shape {rt.shape}")

        P = np.dot(intrinsic,rt3x4)

        pts = list(pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z", "intensity")))

        if not pts:
            return

        try:
            data = np.array(pts, dtype=float)
        except Exception:
            data = np.vstack([np.array(tuple(p), dtype=float) for p in pts])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            return
        xyz = data[:, :3].T  
        if data.shape[1] >= 4:
            intens = data[:, 3].astype(float)
        else:
            intens = np.zeros(xyz.shape[1], dtype=float)

        hom = np.vstack((xyz, np.ones((1, xyz.shape[1]))))  
        proj = np.dot(P,hom)

        z = proj[2, :]
        valid = z > 0
        if not np.any(valid):
            return

        u = np.round(proj[0, valid] / z[valid]).astype(int)
        v = np.round(proj[1, valid] / z[valid]).astype(int)
        cols = undistorted_img.shape[1]
        rows = undistorted_img.shape[0]
        colors = intens[valid].astype(int)

        for ux, vx, col in zip(u, v, colors):
            if 0 <= ux < cols and 0 <= vx < rows:
                c = int(np.clip(col, 0, 255))
                cv2.circle(undistorted_img, (ux, vx), 1, (255, 70, c), -1)

        cv2.imshow("Lidar points on image", undistorted_img)
        cv2.waitKey(1)

def main(args=None):
        rclpy.init(args=args)
        node = Points_on_image()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard Interrupt, shutting down.")
        finally:
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()