#!/usr/bin/env python3

import numpy as np
import json
import yaml

"""CHANGE THE PATH DIRECTORY IN THE MAIN()"""

#just load the data by opening the file as r . Check w3school examples
def load_extrinsic(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    extrinsic_data = data["top_center_lidar-to-cam1-extrinsic"]["param"]["sensor_calib"]["data"]
    print(extrinsic_data)
    extrinsic_matrix = np.array(extrinsic_data)
    print("extrinsic_matrix")
    print(extrinsic_matrix)
    return extrinsic_matrix

def load_intrinsic(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    intrinsic_data = data["camera_matrix"]["data"]
    intrinsic_matrix = np.array(intrinsic_data).reshape(3,3)
    print("intrinsic")
    print(intrinsic_matrix)
    return intrinsic_matrix

def load_points(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

        #find easy method to read and add the points
    
    points3d = [[point['x'], point['y'], point['z']] for point in data["clicked_points"]["3D_points"]]
    points2d = [[point['u'], point['v']] for point in data["clicked_points"]["2D_points"]]
    print (points3d)
    print(points2d)
    #read only 1st 3 points from 3D and 2D ... [:2] 
    return points3d[:3], points2d[:3]  ####Return only the first 3 points

#euclidean should read 2d points. best way to call a function in validation function to which all required data is sent
#this code will perform the validation and calculate error for all the points

def calculate_euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def validation(points3d, intrinsic, R, t, actual_points2d):
    #run loop
    for i in range(len(points3d)):  
        extrinsics = np.dot(R, np.transpose(points3d[i])) + t
        calc_points2d = np.dot(intrinsic, extrinsics)
        
        calc_points2d[0] /= calc_points2d[2]
        calc_points2d[1] /= calc_points2d[2]

        print("The actual point is: %s and the calculated point is : %s" %(actual_points2d[i], calc_points2d[:2]))
        
        error = calculate_euclidean_distance(calc_points2d[:2], actual_points2d[i])
        print("Error corresponding to the point %d: %s" % (i + 1, error))
    return error

def save_params_to_json(file_path,intrinsic_data,extrinsic_data,r,t,error):
    intrinsic_data = intrinsic_data.tolist()
    extrinsic_data = extrinsic_data.tolist()
    r = r.tolist()
    t = t.tolist()
    json_dict = {
        "calculated_error_matrix_camera_to_lidar": {
            "sensor_name": "top_center_lidar",
            "camera_name": "camera1",
            "intrinsic_data": intrinsic_data,
            "extrinsic_data": extrinsic_data,
            "rotation": r,
            "translation":t,
            "error": {
                    "data": error
                }
            }
    }
    with open(file_path, 'w') as file:
        json.dump(json_dict, file, indent=4)

        
if __name__ == '__main__':
    #load the intrinsic, extrinsic, then R and T
    #remember to call the functions to first load the points from json and then send it to vlidate funstion

    intrinsic = load_intrinsic('/home/hitesh/Documents/Project/intrinsic_calibrations_of_camera/intrinsic_cam2.yaml')
    extrinsic_matrix = load_extrinsic('/home/hitesh/Documents/Project/extrinsic_camera calibration_jsonfiles/extrinsic_cam2__.json')  
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]
    print(f"r:",R,"t:",t)

    points3d, actual_points2d = load_points('/home/hitesh/Documents/Project/clicked points/clicked_points_cam2_CALIB.json') 
    error = validation(points3d, intrinsic, R, t, actual_points2d)
    save_params_to_json("/home/hitesh/Documents/Project/error_calibrated_json_files/cam2_to_lidar.json",intrinsic,extrinsic_matrix,R,t,error=error)
