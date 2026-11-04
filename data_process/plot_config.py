# Visualization configuration parameters

# Image dimensions
WIDTH = 640
HEIGHT = 640
WIDTH_E2E = 500
HEIGHT_E2E = 500

CENTER = (int(WIDTH / 2), int(HEIGHT / 2))
CENTER_E2E = (int(WIDTH_E2E / 2), int(HEIGHT_E2E / 2))

# Colors (BGR format)
PT1 = (31, 119, 180)
PT2 = (44, 160, 44)
PT3 = (140, 86, 75)
PT4 = (118, 118, 118)
PT5 = (23, 190, 207)
HEATMAP_COLOR = (127, 127, 127)
EGO_COLOR = (255, 0, 0)         # bgr, blue
TARGET_COLOR = (0, 255, 0)      # bgr, green
OBSTACLE_COLOR = (0, 128, 0)    # bgr, half-green

PT_COLOR = [PT1, PT2, PT3, PT4, PT5]

# Radius for plot
EGO_RADIUS = 15
TARGET_RADIUS = 15
HEATMAP_RADIUS = 15
GEAR_PTS_RADIUS = 15

# Other parameters
SKIP_DISTANCE = 0.5                 # meters, for sequence slice
GEAR_PTS_THRESHOLD_DISTANCE = 1.0   # meters, for gear point slice filtering 
PIXELS_PER_METER_HM = 20            # pixels/meter
PIXELS_PER_METER_E2E = 5            # pixels/meter
MAX_GEAR_PTS = 3                    # max gear points
CLOSE_TO_TARGET_THRESHOLD = 0.6     # meter
YAW_THRESHOLD = 10                  # 10 degree