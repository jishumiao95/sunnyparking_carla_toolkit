import sys
import json
import math
import random
import logging
import pathlib
import os
import copy
import numpy as np
import cv2
import numpy as np
import carla
import pickle
from datetime import datetime
from threading import Thread
from data_generation.obstacles import Obstacle
from data_generation import parking_position
from data_generation.tools import encode_npy_to_pil
from data_generation.world import World
import time

class DataGenerator:
    def __init__(self,script_directory,args):

    
        self._script_directory = script_directory + "/"
        print ("now script directory: "+self._script_directory)
        self._carla_client = carla.Client(args.host, args.port)
        self._carla_client.set_timeout(5.0)

        self._carla_world = self._carla_client.load_world(args.map)
        self._carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # self._seed = args.random_seed
        self._seed = int(time.time())
        
        # random.seed(args.random_seed)
        random.seed(int(time.time()))

        self._world = World(self._carla_world, args)

        # Set initial parking goal index based on collection mode
        if args.collection_mode == 'eval':
            # Evaluation mode: start from 2-1 (index 16)
            self._parking_goal_index = 16 + (2 * args.task_index)
        else:
            # Data collection mode: start from 2-2 (index 17)
            self._parking_goal_index = (2 * args.task_index) % 48 + 17

        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator = parking_position.EgoPosTown04()

        now = datetime.now()
        result_dir = f'_index{args.index}_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self._save_path = pathlib.Path(args.save_path) / args.map / result_dir
        self._save_path.mkdir(parents=True, exist_ok=False)

        self._save_frequency = 3  # save sensor data for every 3 steps 0.1s

        self._num_tasks = args.task_num
        self._task_index = args.task_index
        self._collection_mode = args.collection_mode

        self._distance_diff_to_goal = 10000
        self._rotation_diff_to_goal = 10000
        self._goal_reach_distance_diff = 0.25  # meter
        self._goal_reach_rotation_diff = 2.5 # degree

        # number of frames needs to get into the parking goal in order to consider task completed
        self._num_frames_goal_needed = 2 * 30  # 2s * 30Hz
        self._num_frames_in_goal = 0

        # collected sensor data to output in disk at last
        self._batch_data_frames = []

        self.init()

    @property
    def world(self):
        return self._world

    def world_tick(self):
        self._world.world_tick()

    def render(self, display):
        self._world.render(display)

    def init(self):
        logging.info('*****Init environment for task %d*****', self._task_index)

        # clear all previous setting
        self.destroy()

        # spawn static vehicles in the parking lot
        self._world.init_static_npc(self._seed, self._parking_goal_index)

        # Spawn the player.
        self._ego_transform_generator.update_data_gen_goal_y(self._parking_goal.y)
        ego_transform = self._ego_transform_generator.get_data_gen_ego_transform()

        self._world.init_ego_vehicle(ego_transform)

        # Set up the sensors.
        self._world.init_sensors()

        self._world.next_weather()

        logging.info('*****Init environment for task %d done*****', self._task_index)
        cur_save_path = pathlib.Path(self._save_path) / ('task' + str(self._task_index))
        cur_save_path.mkdir(parents=True, exist_ok=False)
        (cur_save_path / 'replay').mkdir()
        self._carla_client.start_recorder(self._script_directory+str(cur_save_path / 'replay' / 'recoding.log'))
        print("start recorder: " + str(cur_save_path / 'replay' / 'recoding.log'))
        # self.

    def destroy(self):
        self._batch_data_frames.clear()
        self._num_frames_in_goal = 0

        self._world.destroy()

    def soft_destroy(self):
        self._batch_data_frames.clear()
        self._num_frames_in_goal = 0

        self._world.soft_destroy()

    def tick(self, clock):

        self._world.distance_diff_to_goal = self._distance_diff_to_goal
        self._world.rotation_diff_to_goal = self._rotation_diff_to_goal

        is_collision = self._world.tick(clock, self._parking_goal_index)
        if is_collision:
            self.soft_restart()
            return

        # save sensor data in every self.save_frequency frame
        # TODO: check this world step
        step = self._world.step
        if step % self._save_frequency == 0:
            sensor_data_frame = self._world.sensor_data_frame
            sensor_data_frame['bev_state'] = self._world.bev_state
            self._batch_data_frames.append(sensor_data_frame.copy())

        # check if parking_goal is reached
        self.check_goal()

    def check_goal(self):
        t = self._world.ego_transform
        p = t.location
        r = t.rotation

        all_parking_goals = self._world.all_parking_goals

        # find the closest goal
        self._distance_diff_to_goal = sys.float_info.max
        closest_goal = [0.0, 0.0, 0.0]  # (x, y, yaw)
        for parking_goal in all_parking_goals:
            if p.distance(parking_goal) < self._distance_diff_to_goal:
                self._distance_diff_to_goal = p.distance(parking_goal)
                closest_goal[0] = parking_goal.x
                closest_goal[1] = parking_goal.y
                closest_goal[2] = r.yaw

        # find rotation difference
        self._rotation_diff_to_goal = math.sqrt(min(abs(r.yaw), 180 - abs(r.yaw)) ** 2 + r.roll ** 2 + r.pitch ** 2)

        # check if goal is reached
        if self._distance_diff_to_goal < self._goal_reach_distance_diff and \
                self._rotation_diff_to_goal < self._goal_reach_rotation_diff:
            self._num_frames_in_goal += 1
        else:
            self._num_frames_in_goal = 0

        if self._num_frames_in_goal > self._num_frames_goal_needed:
            logging.info('task %d goal reached; ready to save sensor data', self._task_index)
            self.save_sensor_data(closest_goal)
            logging.info('*****task %d done*****', self._task_index)
            self._task_index += 1
            if self._task_index >= self._num_tasks:
                logging.info('completed all tasks; Thank you!')
                exit(0)
            self.restart()

    def restart(self):
        logging.info('*****Config environment for task %d*****', self._task_index)
        # self._carla_client.stop_recorder()
        # clear all previous setting
        self.soft_destroy()

        # Update parking goal index based on collection mode
        if self._collection_mode == 'eval':
            # Evaluation mode: increment by 2 from initial index 16
            self._parking_goal_index = 16 + (2 * self._task_index)
        else:
            # Data collection mode: original logic
            if self._task_index >= 16:
                self._parking_goal_index = 17 + (self._task_index * 2) % 48
            else:
                self._parking_goal_index += 2

        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator.update_data_gen_goal_y(self._parking_goal.y)
        ego_transform = self._ego_transform_generator.get_data_gen_ego_transform()

        self._seed += 1
        self._world.restart(self._seed, self._parking_goal_index, ego_transform)
        current_save_path = pathlib.Path(self._save_path) / ('task' + str(self._task_index))
        (current_save_path / 'replay').mkdir(parents=True, exist_ok=True)
        log_save_path = str(current_save_path / 'replay' / 'recoding.log')
        self._carla_client.start_recorder(log_save_path)
        print("restart recorder: " + log_save_path)
        if os.path.exists(log_save_path):
            os.remove(log_save_path)
            self._carla_client.start_recorder(self._script_directory+log_save_path)

        logging.info('*****Config environment for task %d done*****', self._task_index)

        

    def soft_restart(self):
        logging.info('*****Restart task %d*****', self._task_index)
        self._carla_client.stop_recorder()
        ego_transform = self._ego_transform_generator.get_data_gen_ego_transform()
        # self._world.soft_restart(ego_transform)
        self._world.soft_restart(self._world.spawn_transform)

        # clear cache
        self._batch_data_frames.clear()
        self._num_frames_in_goal = 0
        current_save_path = pathlib.Path(self._save_path) / ('task' + str(self._task_index))
        log_save_path = str(current_save_path / 'replay' / 'recoding.log')
        self._carla_client.stop_recorder()
        if os.path.exists(log_save_path):
            os.remove(log_save_path)
        self._carla_client.start_recorder(self._script_directory+log_save_path)
        print("restart recorder")

        logging.info('*****Restart task %d done*****', self._task_index)

    def save_sensor_data(self, parking_goal):
        # create dirs
        self._carla_client.stop_recorder()
        cur_save_path = pathlib.Path(self._save_path) / ('task' + str(self._task_index))
        cur_save_path.mkdir(parents=True, exist_ok=True)
        (cur_save_path / 'measurements').mkdir()
        (cur_save_path / 'lidar').mkdir()
        (cur_save_path / 'parking_goal').mkdir()
        (cur_save_path / 'topdown').mkdir()
        (cur_save_path / 'parking_layout').mkdir()
        (cur_save_path / 'carla_actor').mkdir()
        for sensor in self._batch_data_frames[0].keys():
            if sensor.startswith('rgb') or sensor.startswith('depth'):
                (cur_save_path / sensor).mkdir()

        total_frames = len(self._batch_data_frames)
        thread_num = 10
        frames_for_thread = total_frames // thread_num
        thread_list = []
        start_index = 0
        end_index = -1
        for t_idx in range(thread_num):
            start = t_idx * frames_for_thread
            start_index = start
            if t_idx == thread_num - 1:
                end_index = total_frames
                end = total_frames
            else:
                end_index = (t_idx + 1) * frames_for_thread
                end = (t_idx + 1) * frames_for_thread
            t = Thread(target=self.save_unit_data, args=(start, end, cur_save_path))
            t.start()
            thread_list.append(t)

        for thread in thread_list:
            thread.join()

        # save Parking Goal
        measurements_file = cur_save_path / 'parking_goal' / '0001.json'
        with open(measurements_file, 'w') as f:
            data = {'x': parking_goal[0],
                    'y': parking_goal[1],
                    'yaw': parking_goal[2]}
            json.dump(data, f, indent=4)

        # save vehicle video
        self._world.save_video(cur_save_path)
        
        # save carla actor
        actor_list_path = str(cur_save_path / 'carla_actor'/'carla_actor.data')
        actor_save_list = []
        parking_layout_obstacles = []
        for actor in self._world._npc_list:
            t_obstacle = Obstacle(actor)
            dict_actor = {"id": t_obstacle.get_actor_id(), "type": t_obstacle.get_actor_type(), "linear_ring": t_obstacle.get_actor_linear_ring(), 
                          "x": t_obstacle.get_actor_transform().location.x,
                          "y":t_obstacle.get_actor_transform().location.y,
                          "yaw":t_obstacle.get_actor_transform().rotation.yaw}
            actor_save_list.append(dict_actor)
            parking_layout_obstacles.append(Obstacle(actor).get_actor_linear_ring())
        with open(actor_list_path, 'wb') as f:
            pickle.dump(actor_save_list, f)
        vehicle_start = self._batch_data_frames[0]['veh_transfrom']
        vehicle_start_pose = (vehicle_start.location.x, vehicle_start.location.y, vehicle_start.rotation.yaw)
        vehicle_end_pose = (parking_goal[0], parking_goal[1], parking_goal[2])
        parking_layout_tuple = (vehicle_start_pose, vehicle_end_pose, parking_layout_obstacles)
        parking_layout_path = str(cur_save_path / 'parking_layout' / 'parking_layout.data')
        with open(parking_layout_path, 'wb') as f:
            pickle.dump(parking_layout_tuple, f)
        logging.info('saved sensor data for task %d in %s', self._task_index, str(cur_save_path))

    def save_unit_data(self, start, end, cur_save_path):

        for index in range(start, end):
            data_frame = self._batch_data_frames[index]

            # save camera / lidar
            for sensor in data_frame.keys():
                if sensor.startswith('rgb'):
                    # _, image = self.image_process(self.target_parking_goal, cam_id=sensor, image=data_frame[sensor])
                    # image = Image.fromarray(image)
                    # image.save(str(cur_save_path / sensor / ('%04d.png' % index)))
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))
                elif sensor.startswith('depth'):
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))
                elif sensor.startswith('lidar'):
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / 'lidar' / ('%04d.ply' % index)))

            # save measurements
            imu_data = data_frame['imu']
            gnss_data = data_frame['gnss']
            vehicle_transform = data_frame['veh_transfrom']
            vehicle_velocity = data_frame['veh_velocity']
            vehicle_control = data_frame['veh_control']

            data = {
                'x': vehicle_transform.location.x,
                'y': vehicle_transform.location.y,
                'z': vehicle_transform.location.z,
                'pitch': vehicle_transform.rotation.pitch,
                'yaw': vehicle_transform.rotation.yaw,
                'roll': vehicle_transform.rotation.roll,
                'speed': (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)),
                'Throttle': vehicle_control.throttle,
                'Steer': vehicle_control.steer,
                'Brake': vehicle_control.brake,
                'Reverse': vehicle_control.reverse,
                'Hand brake': vehicle_control.hand_brake,
                'Manual': vehicle_control.manual_gear_shift,
                'Gear': {-1: 'R', 0: 'N'}.get(vehicle_control.gear, vehicle_control.gear),
                'acc_x': imu_data.accelerometer.x,
                'acc_y': imu_data.accelerometer.y,
                'acc_z': imu_data.accelerometer.z,
                'gyr_x': imu_data.gyroscope.x,
                'gyr_y': imu_data.gyroscope.y,
                'gyr_z': imu_data.gyroscope.z,
                'compass': imu_data.compass,
                'lat': gnss_data.latitude,
                'lon': gnss_data.longitude
            }

            measurements_file = cur_save_path / 'measurements' / ('%04d.json' % index)
            with open(measurements_file, 'w') as f:
                json.dump(data, f, indent=4)

            def save_img(image, keyword=""):
                img_save = np.moveaxis(image, 0, 2)
                save_path = str(cur_save_path / 'topdown' / ('encoded_%04d' % index + keyword + '.png'))
                cv2.imwrite(save_path, img_save)

            keyword = ""
            bev_view1 = self._world.render_BEV_from_state(data_frame['bev_state'])
            img1 = encode_npy_to_pil(np.asarray(bev_view1.squeeze().cpu()))
            save_img(img1, keyword)
