import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import cv2
from tensorboard.compat.tensorflow_stub.io.gfile import isdir
import pickle
from pathlib import Path
import plot_config as cfg

def read_json_files(json_folder):
    coordinates, yaws, gears = [], [], []
    for json_file in sorted(os.listdir(json_folder)):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file), 'r') as f:
                data_point = json.load(f)
                coordinates.append((data_point['x'], data_point['y']))
                yaws.append(data_point['yaw']) # add yaw to check the heading
                
                # if there is 'gear' in data_point, add gear to gear list
                if 'Gear' in data_point:
                    gears.append(data_point['Gear'])
                else:
                    continue

    return coordinates, yaws, gears

def world_move(P_origin, P_slot):
    yaw_0_rad = math.radians(P_origin[2])
    dx = P_slot[0] - P_origin[0]
    dy = P_slot[1] - P_origin[1]

    x_ego = dx * math.cos(yaw_0_rad) + dy * math.sin(yaw_0_rad)
    y_ego = -dx * math.sin(yaw_0_rad) + dy * math.cos(yaw_0_rad)

    return [x_ego, y_ego]

def parse_static_npc(task_path):
    static_npc_position_file = task_path + "/"+ "parking_layout.data" # vehicle info in parking_layout.data
    pos_data = []
    with open(static_npc_position_file, 'rb') as f:
        data = pickle.load(f)
        for obstacle in data[2]:
            obstacle_coords = list(obstacle.coords)
            new_obstacle_coords = [obstacle_coords[0], obstacle_coords[1], obstacle_coords[2], obstacle_coords[3]]
            pos_data.append(new_obstacle_coords)
    return pos_data

def sum_up_gear_times(ego_data, yaw_data, gear_data, target_pos=None):
    flips = 0
    valid_gear_shift = []
    gear_shift_list = []
    
    # Initialize with the first gear value - nothing to compare to yet    
    prev_gear = gear_data[0] if len(gear_data) > 0 else None

    # Start from 1 to avoid invalid index
    for i in range(1, len(gear_data)):
        if len(valid_gear_shift) >= cfg.MAX_GEAR_PTS:
            break
            
        # Check for gear changes
        if gear_data[i] != prev_gear:
            # Skip the initial "N" to "D" transition but keep "N" to "R" transitions
            if prev_gear == "N" and gear_data[i] == "1":
                # Skip this transition but update prev_gear
                prev_gear = gear_data[i]
                continue
                
            current_point = [ego_data[i][0], ego_data[i][1], yaw_data[i]]
            
            if target_pos is not None:
                dx = current_point[0] - target_pos[0]
                dy = current_point[1] - target_pos[1]
                distance_to_target = math.sqrt(dx*dx + dy*dy)
                
                if distance_to_target < cfg.CLOSE_TO_TARGET_THRESHOLD:
                    prev_gear = gear_data[i]  
                    continue
            
            # This is a valid gear shift (including "N" to "R")
            flips += 1
            gear_shift_list.append(current_point) # x, y, yaw
            
            # judgement here
            if len(gear_shift_list) > 1:
                dx = gear_shift_list[-1][0] - gear_shift_list[-2][0]
                dy = gear_shift_list[-1][1] - gear_shift_list[-2][1] 
                
                if math.sqrt(dx*dx + dy*dy) < cfg.GEAR_PTS_THRESHOLD_DISTANCE: 
                    continue
                else:
                    valid_gear_shift.append(gear_shift_list[-2])
                    gear_shift_list = [gear_shift_list[-1]]  
                    
                    if len(valid_gear_shift) >= cfg.MAX_GEAR_PTS:
                        valid_gear_shift.append(gear_shift_list[-1])
                        return valid_gear_shift[:cfg.MAX_GEAR_PTS]
            
            # Update prev_gear after processing
            prev_gear = gear_data[i]
        else:
            prev_gear = gear_data[i]
    
    if gear_shift_list and len(valid_gear_shift) < cfg.MAX_GEAR_PTS:
        if target_pos is not None:
            dx = gear_shift_list[-1][0] - target_pos[0]
            dy = gear_shift_list[-1][1] - target_pos[1]
            distance_to_target = math.sqrt(dx*dx + dy*dy)
            
            if distance_to_target >= cfg.CLOSE_TO_TARGET_THRESHOLD:
                valid_gear_shift.append(gear_shift_list[-1])
        else:
            valid_gear_shift.append(gear_shift_list[-1])
    
    return valid_gear_shift[:cfg.MAX_GEAR_PTS]

def obstacle_plot(background_img, obstacle_list, ego_origin):
    for pos in obstacle_list:
        p1, p2, p3, p4 = pos[0], pos[1], pos[2], pos[3]

        p1_odom = world_move(ego_origin, p1)
        p2_odom = world_move(ego_origin, p2)
        p3_odom = world_move(ego_origin, p3)
        p4_odom = world_move(ego_origin, p4)

        p1_pixel = (int(cfg.CENTER[0] + p1_odom[1] * cfg.PIXELS_PER_METER_HM), 
                    int(cfg.CENTER[1] - p1_odom[0] * cfg.PIXELS_PER_METER_HM))
        p2_pixel = (int(cfg.CENTER[0] + p2_odom[1] * cfg.PIXELS_PER_METER_HM), 
                    int(cfg.CENTER[1] - p2_odom[0] * cfg.PIXELS_PER_METER_HM))
        p3_pixel = (int(cfg.CENTER[0] + p3_odom[1] * cfg.PIXELS_PER_METER_HM), 
                    int(cfg.CENTER[1] - p3_odom[0] * cfg.PIXELS_PER_METER_HM))
        p4_pixel = (int(cfg.CENTER[0] + p4_odom[1] * cfg.PIXELS_PER_METER_HM), 
                    int(cfg.CENTER[1] - p4_odom[0] * cfg.PIXELS_PER_METER_HM))
        pts = np.array([
                            [p1_pixel[0], p1_pixel[1]],
                            [p2_pixel[0], p2_pixel[1]],
                            [p3_pixel[0], p3_pixel[1]],
                            [p4_pixel[0], p4_pixel[1]]
                        ], np.int32)    
        
        cv2.fillPoly(background_img, [pts], color=cfg.OBSTACLE_COLOR)

def gear_pts_plot(gear_image, pos_list, yaw_list, gear_list, target_pos=None):
    ego_origin = [pos_list[0][0], pos_list[0][1], yaw_list[0]]
    gear_points = sum_up_gear_times(pos_list, yaw_list, gear_list, target_pos) # a list of [x, y, yaw]
    
    if not len(gear_points):
        return
    else: 
        for i in range(len(gear_points)): 
            pt = [gear_points[i][0], gear_points[i][1], gear_points[i][2]]
            gear_ego_coordinate = world_move(ego_origin, pt)  # Target slot at VCS

            # gear shifting points at image coordinate. 
            img_gear = (int(cfg.CENTER[0] + gear_ego_coordinate[1] * cfg.PIXELS_PER_METER_HM), 
                        int(cfg.CENTER[1] - gear_ego_coordinate[0] * cfg.PIXELS_PER_METER_HM))
            cv2.circle(gear_image, img_gear, cfg.GEAR_PTS_RADIUS, cfg.PT_COLOR[i], -1)  # ego + target slot

def hm_plot(hm_image, pos_list, yaw_list, ego_origin):
    for i in range(len(pos_list)):
        P_Ego = [pos_list[i][0], pos_list[i][1], yaw_list[i]]

        ego_atEgoCoordi = world_move(ego_origin, P_Ego) # single frame
        path_point_pixel = (int(cfg.CENTER[0] + ego_atEgoCoordi[1] * cfg.PIXELS_PER_METER_HM), 
                            int(cfg.CENTER[1] - ego_atEgoCoordi[0] * cfg.PIXELS_PER_METER_HM))
        
        cv2.circle(hm_image, path_point_pixel, cfg.HEATMAP_RADIUS, cfg.HEATMAP_COLOR, -1)  # gt path

def ego_slot_plot(background_img, ego_origin, slot_origin):
    cv2.circle(background_img, cfg.CENTER, cfg.EGO_RADIUS, cfg.EGO_COLOR, -1) 
    # Target parking slot position
    slot_ego_coordinate = world_move(ego_origin, slot_origin) # Target slot at VCS
    # Target slot at image coordinate
    img_slot_pos = (int(cfg.CENTER[0] + slot_ego_coordinate[1] * cfg.PIXELS_PER_METER_HM), 
                    int(cfg.CENTER[1] - slot_ego_coordinate[0] * cfg.PIXELS_PER_METER_HM))
    # plot ego + target slot    
    cv2.circle(background_img, img_slot_pos, cfg.TARGET_RADIUS, cfg.TARGET_COLOR, -1)  

def check_sequence_in_bounds(pos_list, yaw_list, ego_origin):
    for i in range(len(pos_list)):
        P_Ego = [pos_list[i][0], pos_list[i][1], yaw_list[i]]
        ego_atEgoCoordi = world_move(ego_origin, P_Ego)
        
        pixel_x = int(cfg.CENTER[0] + ego_atEgoCoordi[1] * cfg.PIXELS_PER_METER_HM)
        pixel_y = int(cfg.CENTER[1] - ego_atEgoCoordi[0] * cfg.PIXELS_PER_METER_HM)
        
        if (pixel_x - cfg.HEATMAP_RADIUS < 0 or 
            pixel_x + cfg.HEATMAP_RADIUS >= cfg.WIDTH or 
            pixel_y - cfg.HEATMAP_RADIUS < 0 or 
            pixel_y + cfg.HEATMAP_RADIUS >= cfg.HEIGHT):
            return False
    
    return True

def is_invalid_parking(gear_list, final_yaw, target_yaw):
    if len(set(gear_list)) == 1:
        return True, f"Only used one gear: {gear_list[0]}"
    
    if set(gear_list) == {"N", "1"}:
        try:
            first_one_index = gear_list.index("1")
            if set(gear_list[first_one_index:]) == {"1"}:
                return True, "Only N to 1 transition, no gear shifting"
        except ValueError:
            pass
    
    yaw_diff = (final_yaw - target_yaw + 180) % 360 - 180
    abs_yaw_diff = abs(yaw_diff)
    
    if abs_yaw_diff > cfg.YAW_THRESHOLD:
        return True, f"Final heading differs from target by {abs_yaw_diff:.2f} degrees (> {cfg.YAW_THRESHOLD}°)"
    
    return False, ""

def main():
    
    name_list = [
        "00_Zhu", "01_Yuki", "02_Gia", "03_Takahashi", "04_Miao",
        "05_Seo", "06_Shimomura", "07_Nic", "08_Wang",
    ]

    for name in name_list:
        source_path = '/path/to/dataset/carla_parking/' + name
        save_path = '/path/to/dataset/generated/'
        scene_name = "carla_town04_"
        timestamp = "0318"
        
        for log_timestamp in os.listdir(source_path):    
            log_file_path = os.path.join(source_path, log_timestamp)
            if not isdir(log_file_path):
                continue
            
            seq_path = os.path.join(save_path, scene_name+timestamp, name, log_timestamp) 
            bad_case_path = os.path.join(save_path, scene_name+timestamp, name, log_timestamp)
            
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
                print(f"Make dir successfully: {seq_path} \n")
        
            for log_file in os.listdir(log_file_path):
                
                log_path = os.path.join(log_file_path, log_file)
                # gen bev pic
                target_slot_path = os.path.join(log_path, "parking_goal/")
                ego_trajectory_path = os.path.join(log_path, "measurements/")
                obstacle_path = os.path.join(log_path, "parking_layout/")

                ego_coordinates, ego_yaws, ego_gears = read_json_files(ego_trajectory_path)
                target_coordinates, target_yaws, _ = read_json_files(target_slot_path)
                static_data = parse_static_npc(obstacle_path)

                is_invalid, invalid_reason = is_invalid_parking(ego_gears, ego_yaws[-1], target_yaws[0])
                
                if is_invalid:
                    print(f"====> Invalid parking case detected: {log_file} - {invalid_reason}")
                    # bad_case_seq_path = os.path.join(bad_case_path, scene_name+timestamp, name, log_timestamp, log_file)
                    save_input_path = Path(bad_case_path, f"{log_file}/inputs/")
                    save_viz_path = Path(bad_case_path, f"{log_file}/viz/")
                    save_gsp_gt_path = Path(bad_case_path, f"{log_file}/gt_gsp/")
                    save_info_path = Path(bad_case_path, f"{log_file}/")
                    
                    save_input_path.mkdir(parents=True, exist_ok=True)
                    save_viz_path.mkdir(parents=True, exist_ok=True)
                    save_gsp_gt_path.mkdir(parents=True, exist_ok=True)

                    
                    with open(os.path.join(save_info_path, "invalid_reason.txt"), "w") as f:
                        f.write(invalid_reason)

                    case_info = {
                        "reason": invalid_reason,
                        "log_file": log_file,
                        "timestamp": log_timestamp,
                        "total_frames": len(ego_coordinates),
                        "gear_sequence": ego_gears,
                        "target_position": target_coordinates[0]
                    }

                    with open(os.path.join(save_info_path, "case_details.json"), "w") as f:
                        json.dump(case_info, f, indent=4)
                else:
                    save_input_path = Path(seq_path, f"{log_file}/inputs/")
                    save_viz_path = Path(seq_path, f"{log_file}/viz/")
                    save_gsp_gt_path = Path(seq_path, f"{log_file}/gt_gsp/")
                    
                    save_input_path.mkdir(parents=True, exist_ok=True)
                    save_viz_path.mkdir(parents=True, exist_ok=True)
                    save_gsp_gt_path.mkdir(parents=True, exist_ok=True)
                
                if len(ego_coordinates) != len(ego_yaws) or len(ego_coordinates) != len(ego_gears):
                    raise ValueError("ego_coordinates, ego_yaws and ego_gears are not the same length")

                prev_min_index = None
                
                for seq_i in range(len(ego_coordinates)):
                    # a for-loop for e2e_bev
                    # ego position at t_0 
                    ego_origin = [ego_coordinates[seq_i][0], ego_coordinates[seq_i][1], ego_yaws[seq_i]]
                    slot_origin = [target_coordinates[0][0], target_coordinates[0][1], target_yaws[0]]
                    
                    egoPos_seq = ego_coordinates[seq_i:]
                    egoYaw_seq = ego_yaws[seq_i:]
                    egoGear_seq = ego_gears[seq_i:]
                    
                    if not check_sequence_in_bounds(egoPos_seq, egoYaw_seq, ego_origin):
                        print(f"====> Sequence {seq_i} contains points outside image bounds, skipping.")
                        continue
                    
                    back_seq_image = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3), dtype=np.uint8)       # 640, 640
                    gt_seq_image = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3), dtype=np.uint8)         # 640, 640
                    gt_seq_gear_image = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3), dtype=np.uint8)    # 640, 640
                    
                    ego_slot_plot(back_seq_image, ego_origin, slot_origin)
                    
                    # Skip if movement is less than [skip_distance] from previous frame
                    if seq_i > 0 and seq_i != len(ego_coordinates)-1:
                        if prev_min_index is None:
                            prev_pos = ego_coordinates[seq_i-1]
                            current_pos = ego_coordinates[seq_i]

                            dx = current_pos[0] - prev_pos[0]
                            dy = current_pos[1] - prev_pos[1]
                            if math.sqrt(dx*dx + dy*dy) < cfg.SKIP_DISTANCE:
                                prev_min_index = seq_i-1
                                # print(f"====> invalid. pass")
                                continue
                        else: 
                            prev_pos = ego_coordinates[prev_min_index]
                            current_pos = ego_coordinates[seq_i]
                            dx = current_pos[0] - prev_pos[0]
                            dy = current_pos[1] - prev_pos[1]
                            if math.sqrt(dx*dx + dy*dy) < cfg.SKIP_DISTANCE:
                                # print(f"====> invalid. pass")
                                continue
                            else: 
                                prev_min_index = None

                    hm_plot(gt_seq_image, egoPos_seq, egoYaw_seq, ego_origin)
                    target_pos = (target_coordinates[0][0], target_coordinates[0][1])
                    gear_pts_plot(gt_seq_gear_image, egoPos_seq, egoYaw_seq, egoGear_seq, target_pos) # size: 640. 640
                    obstacle_plot(back_seq_image, static_data, ego_origin)
                
                    ######################## DO NOT CHANGE THIS PART #################################
                    gt_seq_mix_image = np.zeros_like(gt_seq_image)
                    gt_seq_mix_image[:,:] = gt_seq_image
                    gt_seq_mix_image = np.where(gt_seq_gear_image != 0, gt_seq_gear_image, gt_seq_image)
        
                    viz_seq_image = np.zeros_like(back_seq_image)
                    viz_seq_image[:,:] = gt_seq_mix_image
                    viz_seq_image = np.where(back_seq_image != 0, back_seq_image, gt_seq_mix_image)
                    viz_seq_debug = np.hstack((back_seq_image, gt_seq_mix_image,viz_seq_image))
                
                    
                    filename = "encoded_" + str(seq_i).zfill(4) + '.png'
                    cv2.imwrite(str(save_gsp_gt_path/filename), gt_seq_mix_image)
                    cv2.imwrite(str(save_viz_path/filename), viz_seq_debug)
                    cv2.imwrite(str(save_input_path/filename), back_seq_image)
                print(f"=====> Successfully save sequence at {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processer from CARLA parking')
    parser.add_argument('--path', type=str, help="Filename setting")

    args = parser.parse_args()
    main()