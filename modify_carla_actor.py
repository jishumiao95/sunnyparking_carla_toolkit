import pickle
import argparse
from pathlib import Path
import json
import math
from shapely.geometry import LinearRing, Point
def read_e2e_data(args):
    data_list = []
    current_dir = Path(__file__).parent
    sub_folder = args.folder
    data_dir = current_dir / args.dataset / sub_folder
    i = 0
    for file in data_dir.glob("*/*/carla_actor/carla_actor.data"):
        # print(new_data)
        with open(file, "rb") as f:
            i+=1
            data = pickle.load(f)
            for actor_json in data:
                t_linear_ring = list(actor_json["linear_ring"].coords)
                n_linear_ring = [t_linear_ring[0], t_linear_ring[1], t_linear_ring[3], t_linear_ring[2], t_linear_ring[4]]
                actor_json["linear_ring"] = LinearRing(n_linear_ring)
            with open(file, "wb") as f:
                pickle.dump(data,f)
            # data_begin_rad = (data[0][0], data[0][1], math.radians(data[0][2]))
            # data_dest_rad = (data[1][0], data[1][1], math.radians(data[1][2]))
            # new_data = (data_begin_rad, data_dest_rad, data[2])
            # print(data)
        print(f"task{i}:"+ "*"*20)
    # return data_list
        # print(f"task{i}:"+ "*"*20)
            # data_list.append(data)
    # return data_list
    
def main(): 
    argparser = argparse.ArgumentParser(
        description='CARLA Data Reader')
    argparser.add_argument(
        '--folder',
        metavar='F',
        default='Town04_Opt/driver', #Your dir
        help='folder name of sub folder')
    argparser.add_argument(
        '--dataset',
        metavar='E',
        default='e2e_parking',
        help='folder name of dataset folder')
    args = argparser.parse_args()
    # read_e2e_data(args)
    read_e2e_data(args)
    # with open(f"{args.folder}.data", "wb") as f:
    #     pickle.dump(data_list, f)
    print(f"Saved {args.folder}.data")


    """
    <LINEARRING (274.928 -197.014, 274.928 -199.046, 270.072 -197.014, 270.072 -...>, 
    <LINEARRING (300.403 -206.315, 300.403 -208.285, 296.597 -206.315, 296.597 -...>,
    """

if __name__ == "__main__":
    main()