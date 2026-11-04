import pickle
import argparse
from pathlib import Path
import json
import math
from shapely.geometry import LinearRing, Point
def read_e2e_data(args):
    current_dir = Path(__file__).parent
    sub_folder = args.folder
    data_dir = current_dir / args.dataset / sub_folder
    i = 0
    for file in data_dir.glob("*/*/"):
        file_measurements = file / "measurements"/"0000.json"
        with open(file_measurements, "r") as f:
            measurements = json.load(f)
        start_pose = (float(measurements["x"]), float(measurements["y"]), float(measurements["yaw"]))
        file_data = file / "parking_layout" / "parking_layout.data"
        with open(file_data, "rb") as f:
            i+=1
            data = pickle.load(f)
            new_obstacles = [] 
            # new_data_start = (data[0][0], data[0][1],math.degrees(data[0][2]))
            new_data_end = (data[1][0], data[1][1],data[1][2])
            for obstacle in data[2]:
                obstacle_coords = list(obstacle.coords)
                new_obstacle_coords = [obstacle_coords[0], obstacle_coords[1], obstacle_coords[3], obstacle_coords[2], obstacle_coords[0]]
                new_obstacles.append(LinearRing(new_obstacle_coords))
        new_data = (start_pose, new_data_end, new_obstacles)
        with open(file_data, "wb") as f:
            pickle.dump(new_data, f)
        # print(new_data)
        print(f"task{i}:"+ "*"*20)
            # data_list.append(data)
    # return data_list
    
def main(): 
    argparser = argparse.ArgumentParser(
        description='CARLA Data Reader')
    argparser.add_argument(
        '--folder',
        metavar='F',
        default='Town04_Opt/driver', # Your dir
        help='folder name of sub folder')
    argparser.add_argument(
        '--dataset',
        metavar='E',
        default='e2e_parking',
        help='folder name of dataset folder')
    args = argparser.parse_args()
    read_e2e_data(args)
    # data_list = read_e2e_data(args)
    # with open(f"{args.folder}.data", "wb") as f:
    #     pickle.dump(data_list, f)
    # print(f"Saved {args.folder}.data")


    """
    <LINEARRING (274.928 -197.014, 274.928 -199.046, 270.072 -197.014, 270.072 -...>, 
    <LINEARRING (300.403 -206.315, 300.403 -208.285, 296.597 -206.315, 296.597 -...>,
    """

if __name__ == "__main__":
    main()