import argparse
import logging
import carla
from pathlib import Path
import pygame

from data_generation.data_generator import DataGenerator
from data_generation.keyboard_control import KeyboardControl

script_path = Path(__file__).resolve()
script_directory = str(script_path.parent)
def game_loop(args):
    pygame.init()
    pygame.font.init()
    data_generator = None

    try:
        logging.info('Load Map %s', args.map)
        data_generator = DataGenerator(script_directory, args)
        controller = KeyboardControl(data_generator.world)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        clock = pygame.time.Clock()
        while True:
            data_generator.world_tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(data_generator._carla_client, data_generator.world, clock):
                return
            data_generator.tick(clock)
            data_generator.render(display)
            pygame.display.flip()

    finally:
        if data_generator is not None:
            data_generator.destroy()

        pygame.quit()

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Data Generation')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='860x480',
        help='window resolution (default: 860x480)')
    argparser.add_argument(
        '--gamma',
        default=0.0,
        type=float,
        help='Gamma correction of the camera (default: 0.0)')
    argparser.add_argument(
        '--save_path',
        default='./e2e_parking/',
        help='path to save sensor data (default: ./e2e_parking/)')
    argparser.add_argument(
        '--task_num',
        default=16,
        type=int,
        help='number of parking task (default: 16')
    argparser.add_argument(
        '--steering_angle',
        default=30,
        type=int,
        help='max steering angle of front wheel (default: 30)')
    argparser.add_argument(
        '--map',
        default='Town04_Opt',
        help='map of carla (default: Town04_Opt)',
        choices=['Town04_Opt', 'Town05_Opt'])
    argparser.add_argument(
        '--shuffle_veh',
        default=True,
        type=str2bool,
        help='shuffle static vehicles between tasks (default: True)')
    argparser.add_argument(
        '--shuffle_weather',
        default=False,
        type=str2bool,
        help='shuffle weather between tasks (default: False)')
    argparser.add_argument(
        '--weather_index',
        default=None,
        type=int,
        help='initial weather index (0=ClearNoon, 1=ClearSunset, 2=CloudyNoon, 3=CloudySunset, etc.). If not set, uses default (0)')
        
        
    argparser.add_argument(
        '--random_seed',
        default=0,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)')
    argparser.add_argument(
        '--bev_render_device',
        default='cuda',
        help='device used for BEV Rendering (default: cpu)',
        choices=['cpu', 'cuda'])
    argparser.add_argument(
        '--x_range',
        default = 1,
        type = float,
        help = 'generalization range of x [-val,val]'
    )
    argparser.add_argument(
        '--y_range',
        default = 10,
        type = float,
        help = 'generalization range of y [-val,val]'
    )
    argparser.add_argument(
        '--x_step',
        default = 1,
        type = float,
        help = 'generalization of x step (meter)'
    )
    argparser.add_argument(
        '--y_step',
        default = 2,
        type = float,
        help = 'generalization of y step (meter)'
    )
    argparser.add_argument(
        '--index',
        default = 0,
        type = int ,
        help = 'generalization index of x & y in the same task'
    )
    argparser.add_argument(
        '--task_index',
        default = 0,
        type = int ,
        help = 'begin from task index (0-15 for data collection mode, 0-15 for evaluation mode)'
    )
    argparser.add_argument(
        '--collection_mode',
        default = 'data',
        type = str,
        choices = ['data', 'eval'],
        help = 'data collection mode: "data" for original data collection indices (2-2, 2-4, etc.), "eval" for evaluation indices (2-1, 2-3, etc.)'
    )
    argparser.add_argument(
        '--target_fps',
        default = 30,
        type = int,
        help = 'target frames per second for simulation (default: 30)'
    )
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')

if __name__ == '__main__':
    main()
