import argparse
import os
import json
import math
import numpy as np
import cv2
from tensorboard.compat.tensorflow_stub.io.gfile import isdir
import pickle
from pathlib import Path
import plot_config as cfg


def read_json_files(json_folder):
    coordinates, yaws = [], []
    for json_file in sorted(os.listdir(json_folder)):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file), 'r') as f:
                data_point = json.load(f)
                coordinates.append((data_point['x'], data_point['y']))
                yaws.append(data_point['yaw']) # add yaw to check the heading

    return coordinates, yaws


def world_move(P_origin, P_slot):
    # 将角度转换为弧度
    yaw_0_rad = math.radians(P_origin[2])
    # yaw_0_rad = math.radians(90)
    # world 坐标系转为 ego 坐标系
    dx = P_slot[0] - P_origin[0]
    dy = P_slot[1] - P_origin[1]

    # 旋转变换
    x_ego = dx * math.cos(yaw_0_rad) + dy * math.sin(yaw_0_rad)
    y_ego = -dx * math.sin(yaw_0_rad) + dy * math.cos(yaw_0_rad)

    return [x_ego, y_ego]

def parse_static_npc(task_path):
    static_npc_position_file = task_path + "/"+ "parking_layout.data"
    pos_data = []
    with open(static_npc_position_file, 'rb') as f:
        data = pickle.load(f)
        for obstacle in data[2]:
            obstacle_coords = list(obstacle.coords)
            new_obstacle_coords = [obstacle_coords[0], obstacle_coords[1], obstacle_coords[2], obstacle_coords[3]]
            pos_data.append(new_obstacle_coords)
    return pos_data



def main():
    name_list = ["00_Zhu", "01_Yuki", "02_James", "03_Takahashi", "04_Miao", "05_Seo", "06_Shimomura", "07_Oliver", "08_Wang"]
    
    for name in name_list:

        source_path = '/workspace/dataset/carla_parking_human_mar19/'+ name # docker container path
    
        for log_timestamp in os.listdir(source_path):    
            
            log_file_path = os.path.join(source_path, log_timestamp)
            if not isdir(log_file_path):
                continue
        
            for task_index in os.listdir(log_file_path):
                print(f"log file path: {log_file_path}")
                
                log_path = os.path.join(log_file_path, task_index)

                # gen bev pic
                ego_trajectory_path = os.path.join(log_path, "measurements/")
                obstacle_path = os.path.join(log_path, "parking_layout/")

                ego_coordinates, ego_yaws = read_json_files(ego_trajectory_path)  # 读取轨迹path点集，[x,y], yaw
                static_data = parse_static_npc(obstacle_path)

                
                for i in range(len(ego_coordinates)):
                    print(i)
                    # ego position at t_0 
                    ego_origin = [ego_coordinates[i][0], ego_coordinates[i][1], ego_yaws[i]]
                    back_e2e_image = np.zeros((cfg.HEIGHT_E2E, cfg.WIDTH_E2E, 3), dtype=np.uint8) # 500. 500

                    # draw static
                    for pos in static_data:
                        p1 = pos[0]
                        p2 = pos[1]
                        p3 = pos[2]
                        p4 = pos[3]

                        p1_odom = world_move(ego_origin, p1)
                        p2_odom = world_move(ego_origin, p2)
                        p3_odom = world_move(ego_origin, p3)
                        p4_odom = world_move(ego_origin, p4)
                        
                        
                        p1_e2e_pixel = (int(cfg.CENTER_E2E[0] + p1_odom[1] * cfg.PIXELS_PER_METER_E2E), 
                                        int(cfg.CENTER_E2E[1] - p1_odom[0] * cfg.PIXELS_PER_METER_E2E))
                        p2_e2e_pixel = (int(cfg.CENTER_E2E[0] + p2_odom[1] * cfg.PIXELS_PER_METER_E2E), 
                                        int(cfg.CENTER_E2E[1] - p2_odom[0] * cfg.PIXELS_PER_METER_E2E))
                        p3_e2e_pixel = (int(cfg.CENTER_E2E[0] + p3_odom[1] * cfg.PIXELS_PER_METER_E2E), 
                                        int(cfg.CENTER_E2E[1] - p3_odom[0] * cfg.PIXELS_PER_METER_E2E))
                        p4_e2e_pixel = (int(cfg.CENTER_E2E[0] + p4_odom[1] * cfg.PIXELS_PER_METER_E2E), 
                                        int(cfg.CENTER_E2E[1] - p4_odom[0] * cfg.PIXELS_PER_METER_E2E))
                        
                        pts_e2e = np.array([
                            [p1_e2e_pixel[0], p1_e2e_pixel[1]],
                            [p2_e2e_pixel[0], p2_e2e_pixel[1]],
                            [p3_e2e_pixel[0], p3_e2e_pixel[1]],
                            [p4_e2e_pixel[0], p4_e2e_pixel[1]]
                        ], np.int32)
                        
                        cv2.fillPoly(back_e2e_image, [pts_e2e], color=cfg.OBSTACLE_COLOR)
                
                    ######################## DO NOT CHANGE THIS PART #################################
                    filename = "encoded_" + str(i).zfill(4) + '.png'
                    # pdb.set_trace()
                    save_path = Path(log_path, "topdown/")
        
                    save_path.mkdir(parents=True, exist_ok=True)
                
                    cv2.imwrite(str(save_path/filename), back_e2e_image)
                    print(f"=====> Successfully save: {str(save_path/filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processer from CARLA parking')
    parser.add_argument('--path', type=str, help="Filename setting")

    args = parser.parse_args()
    main()