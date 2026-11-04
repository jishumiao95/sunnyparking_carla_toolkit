# Data process readme
Ths manual is a guideline for processing the human parking dataset using E2E Parking data collector 'carla_data_gen.py'. 

## Step-by-step


### Step 1 Heatmap dataset


```Shell
heatmap_img_gen.py
```
- Set the 'source_path' and 'root_path'.

### Step 2 E2E parking dataset

```Shell
e2e_bev_img_gen.py 
```
- Set the 'source_path', the BEV image will be generated to 'topdown' folder inside the directory. 

### Step 3 Folder move

```Shell
file_auto_process.py        
```


## File Structure 

### Source directory path 
Example: 
source_path = $ROOT_PATH/e2e_parking/Town04/Opt + {name}

```
source_path 
├── 03_index0_11_index0_10_index0_48_index0_42 # TODO: known bad episode
│   ├── task0
│   │   ├── carla_actor
│   │   ├── depth_front
│   │   ├── depth_left
│   │   ├── depth_rear
│   │   ├── depth_right
│   │   ├── parking_layout # start/goal + static footprints
│   │   ├── lidar
│   │   ├── measurements # chassis
│   │   ├── parking_goal # target slot
│   │   ├── replay
│   │   ├── rgb_front
│   │   ├── rgb_left
│   │   ├── rgb_rear
│   │   ├── rgb_right
│   │   └── topdown # E2E BEV (be generated later by 'e2e_bev_img_gen.py')
│   └── task1
│   │   └── ...
│   └── taskn
│       └── ...
├── ...
```

### Target directory path 
root_path = $TARGET_PATH/generated/$TIMESTAMP + {name}

```
root_path
├── $DRIVER
│   ├── 02_25_14_29_44
│   │   ├── task0
│   │   │   ├── gt_gsp # gt
│   │   │   ├── inputs # images for network input
│   │   │   └── viz
│   │   ├── task1
│   │   │   ├── gt_gsp
│   │   │   ├── inputs
│   │   │   └── viz
│   │   ├── task10
│   │   │   ├── gt_gsp
│   │   │   ├── inputs
│   │   │   └── viz
│   │   ├── task11
│   │   │   ├── gt_gsp
│   │   │   ├── inputs
│   │   │   └── viz
```

