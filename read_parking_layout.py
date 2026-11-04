import pickle
import argparse
from pathlib import Path
import json
import math
from shapely.geometry import LinearRing, Point

WHEEL_BASE = 2.8
FRONT_HANG = 0.96
REAR_HANG = 0.93
LENGTH = WHEEL_BASE + FRONT_HANG + REAR_HANG
WIDTH = 1.94



def compute_rear_center(center,yaw):
    x, y = center
    # Calculate the distance to move back to the rear center
    distance_to_rear_center = (LENGTH/2.0 - REAR_HANG)  # Move back from the center to the rear center

    # Calculate rear center coordinates
    rear_x = x - distance_to_rear_center * math.cos(yaw)
    rear_y = y - distance_to_rear_center * math.sin(yaw)
    
    return (rear_x, rear_y)



def read_e2e_data(args):
    data_list = []
    current_dir = Path(__file__).parent
    sub_folder = args.folder
    data_dir = current_dir / args.dataset / sub_folder
    for file in data_dir.glob("*/*/parking_layout/parking_layout.data"):
        # print(new_data)
        with open(file, "rb") as f:
            data = pickle.load(f)
            begin_yaw = math.radians(data[0][2])
            b_x,b_y = compute_rear_center((data[0][0],data[0][1]), begin_yaw)
            end_yaw = math.radians(data[1][2])
            d_x,d_y = compute_rear_center((data[1][0],data[1][1]), end_yaw)
            
            data_begin_hope = (b_x, b_y, begin_yaw)
            data_dest_hope = (d_x, d_y, end_yaw)
            
            new_data = (data_begin_hope, data_dest_hope, data[2])
            data_list.append(new_data)
    return data_list
        # print(f"task{i}:"+ "*"*20)
            # data_list.append(data)
    # return data_list
    
def main(): 
    argparser = argparse.ArgumentParser(
        description='CARLA Data Reader')
    argparser.add_argument(
        '--folder',
        metavar='F',
        default='ch',
        help='folder name of sub folder')
    argparser.add_argument(
        '--dataset',
        metavar='E',
        default='e2e',
        help='folder name of sub folder')
    args = argparser.parse_args()
    # read_e2e_data(args)
    data_list = read_e2e_data(args)
    with open(f"{args.folder}.data", "wb") as f:
        pickle.dump(data_list, f)
    print(f"Saved {args.folder}.data")


    """
    <LINEARRING (274.928 -197.014, 274.928 -199.046, 270.072 -197.014, 270.072 -...>, 
    <LINEARRING (300.403 -206.315, 300.403 -208.285, 296.597 -206.315, 296.597 -...>,
    """

if __name__ == "__main__":
    main()