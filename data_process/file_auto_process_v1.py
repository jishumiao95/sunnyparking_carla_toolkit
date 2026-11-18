import shutil
from pathlib import Path
import pdb

val_sample_ratio = 0.1  # Adjust this number by requirements

def copy_seq_images(source_path, destination_path):
    
    global name, scene_name
    # Create destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read all folders in path A
    for folder in source_path.iterdir():
        # Skip if not a directory
        if not folder.is_dir():
            continue
            
        # Step 2: Look for 'input' folder inside each folder
        for task in folder.iterdir(): # task_X

            input_path = task / 'inputs'
            gt_path = task / 'gt_gsp'
            if input_path.exists():
                # Step 3: Copy all PNG files to destination
                i = 0
                for file in input_path.iterdir():
                    if file.suffix.lower() == '.png':
                        new_filename = f"{scene_name}_{name}_{folder.name}_{task.name}_{file.name}"
                        dest_file = destination_path / "images" / new_filename 
                        shutil.copy2(file, dest_file)
                        i+=1
                        print(f"Copied: {file.name}, count: {i}")
            
            if gt_path.exists():
                i = 0
                # Step 3: Copy all PNG files to destination
                for file in gt_path.iterdir():
                    if file.suffix.lower() == '.png':
                        new_filename = f"{scene_name}_{name}_{folder.name}_{task.name}_{file.name}"
                        dest_file = destination_path / "head1" / new_filename
                        shutil.copy2(file, dest_file)
                        i+=1
                        print(f"Copied: {file.name}, count: {i}")

def copy_input_images(source_path, destination_path):
    global name, scene_name

    # Create destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read all folders in path A
    for folder in source_path.iterdir():
        # Skip if not a directory
        if not folder.is_dir():
            continue
            
        # Step 2: Look for 'input' folder inside each folder
        input_path = folder / 'input'
        gt_path = folder / 'gt'
        if input_path.exists():
            # Step 3: Copy all PNG files to destination
            i = 0
            for file in input_path.iterdir():
                if file.suffix.lower() == '.png':
                    new_filename = f"{scene_name}_{name}_{file.name}"
                    dest_file = destination_path / "images" / new_filename # TODO: change the filename here
                    shutil.copy2(file, dest_file)
                    i+=1
                    print(f"Copied: {file.name}, count: {i}")
        
        if gt_path.exists():
            i = 0
            # Step 3: Copy all PNG files to destination
            for file in gt_path.iterdir():
                if file.suffix.lower() == '.png':
                    new_filename = f"{scene_name}_{name}_{file.name}"
                    dest_file = destination_path / "head1" / new_filename
                    shutil.copy2(file, dest_file)
                    i+=1
                    print(f"Copied: {file.name}, count: {i}")

def divide_train_val(destination_path, val_sample_ratio):
    """
    Divide files into training and validation sets
    Args:
        input_path: Path to input directory containing input images
        gt_path: Path to ground truth directory
        image_train: Path for training input images
        image_val: Path for validation input images
        head1_train: Path for training ground truth images
        head1_val: Path for validation ground truth images
        val_sample_size: Number of samples for validation set
    """

    # New divide function paths
    input_path = destination_path/"images"  # Source input folder
    gt_path = destination_path/"head1"    # Source gt folder TODO: support for head2 
    image_train = destination_path/"images/train"  # Destination for training input images
    image_val = destination_path/"images/val"      # Destination for validation input images
    head1_train = destination_path/"head1/train"  # Destination for training ground truth
    head1_val = destination_path/"head1/val"      # Destination for validation ground truth
    
    # Create all necessary directories
    for path in [input_path, gt_path, image_train, image_val, head1_train, head1_val]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Get list of all PNG files in input_path
    png_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() == '.png'])
    total_files = len(png_files)
    
    print(f"total images: {total_files}")

    # Calculate step size for validation set
    val_sample_size = int(total_files * val_sample_ratio)

    step = total_files // val_sample_size
    
    # Track input files and gt files to be deleted after copying
    input_files_to_delete = []
    gt_files_to_delete = []
    
    # Move files to respective directories
    for idx, input_file in enumerate(png_files):
        # Construct corresponding gt file path
        gt_file = gt_path / input_file.name
        
        if not gt_file.exists():
            print(f"Warning: No matching GT file for {input_file.name}")
            continue
        
        # If index is multiple of step, move to validation set
        if idx % step == 0 and len(list(image_val.glob('*.png'))) < val_sample_size:
            shutil.copy2(str(input_file), str(image_val / input_file.name))
            shutil.copy2(str(gt_file), str(head1_val / gt_file.name))
            print(f"Copied to validation set: {input_file.name}")
        else:
            # Move to training set
            shutil.copy2(str(input_file), str(image_train / input_file.name)) 
            shutil.copy2(str(gt_file), str(head1_train / gt_file.name)) 
            print(f"Copied to training set: {input_file.name}") 
        
        # Add files to deletion lists
        input_files_to_delete.append(input_file)
        gt_files_to_delete.append(gt_file)
    
    # Delete original files from parent directories after copying
    print("\nDeleting original files from parent directories...")
    
    # Delete input images
    for file in input_files_to_delete:
        if file.exists() and (file.parent == input_path):
            file.unlink()  # Delete the file
    
    # Delete gt images
    for file in gt_files_to_delete:
        if file.exists() and (file.parent == gt_path):
            file.unlink()  # Delete the file
    
    print(f"Deleted {len(input_files_to_delete)} input files and {len(gt_files_to_delete)} GT files from parent directories.")

def copy_selected_seq_to_destination(source_path, root_path, level):
    """
    Copy sequence data from source to destination with frame filtering based on level.
    
    Args:
        source_path (Path): Source directory path containing driver data
        root_path (Path): Root destination path
        level (str): Level number to determine frames to remove
    """
    # Create destination directory if it doesn't exist
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Define frames to remove based on level only
    frames_to_remove = {
        "1": 10,  # Level 1: remove 0 frames
        "2": 15,  # Level 2: remove 5 frames
        "3": 20  # Level 3: remove 10 frames
    }
    
    n_frames_to_remove = frames_to_remove[level]
    
    for folder in source_path.iterdir():
        if not folder.is_dir():
            continue
            
        # Create destination paths
        head1_dest_path = Path(root_path, "head1", folder.name)
        head2_dest_path = Path(root_path, "head2", folder.name)
        
        # Remove existing directories if they exist
        
        # Get source paths for head1 and head2
        head_1 = Path(folder, "1")
        head_2 = Path(folder, "2")
        
        if head_1.exists():
            # Copy head1 data with frame filtering
            copy_head_data(head_1, head1_dest_path, n_frames_to_remove)
            
        if head_2.exists():
            # Copy head2 data with frame filtering
            copy_head_data(head_2, head2_dest_path, n_frames_to_remove)

def copy_head_data(source_path, dest_path, n_frames_to_remove):
    """
    Copy head data with frame filtering while maintaining folder structure.
    
    Args:
        source_path (Path): Source directory path
        dest_path (Path): Destination directory path
        n_frames_to_remove (int): Number of frames to remove from the end
    """
    # Get all task folders
    # pdb.set_trace() 
    task_folders = sorted([f for f in source_path.iterdir() if f.is_dir() and f.name.startswith("task")])
    
    for task_folder in task_folders:
        # Create corresponding task folder in destination
        task_dest = dest_path / task_folder.name
        task_dest.mkdir(parents=True, exist_ok=True)
        
        # Get all PNG files from input, gt, and viz directories for this task
        input_files = sorted([f for f in (task_folder / "inputs").glob("*.png")])
        gt_files = sorted([f for f in (task_folder / "gt_gsp").glob("*.png")])
        viz_files = sorted([f for f in (task_folder / "viz").glob("*.png")])
        
        # Filter out the last n frames from all lists
        input_files = input_files[:-n_frames_to_remove] if n_frames_to_remove > 0 else input_files
        gt_files = gt_files[:-n_frames_to_remove] if n_frames_to_remove > 0 else gt_files
        viz_files = viz_files[:-n_frames_to_remove] if n_frames_to_remove > 0 else viz_files
        
        # Create subdirectories
        (task_dest / "inputs").mkdir(parents=True, exist_ok=True)
        (task_dest / "gt_gsp").mkdir(parents=True, exist_ok=True)
        (task_dest / "viz").mkdir(parents=True, exist_ok=True)
        
        # Copy filtered files while maintaining synchronization
        for input_file, gt_file, viz_file in zip(input_files, gt_files, viz_files):
            # Copy input file
            input_dest = task_dest / "inputs" / input_file.name
            shutil.copy2(input_file, input_dest)
            
            # Copy gt file
            gt_dest = task_dest / "gt_gsp" / gt_file.name
            shutil.copy2(gt_file, gt_dest)
            
            # Copy viz file
            viz_dest = task_dest / "viz" / viz_file.name
            shutil.copy2(viz_file, viz_dest)
            
            print(f"Copied from {task_folder.name}: {input_file.name}, {gt_file.name}, {viz_file.name}")

# def remove_frames(source_path, root_path):

if __name__ == "__main__":

    head_list = ["1", "2"]
    driver_list = {
        "00_Zhu": "1",
        "01_Yuki": "3",
        "02_Gia": "2",
        "03_Takahashi": "2",
        "04_Miao": "1",
        "05_Seo": "2",
        "06_Shimomura": "2",
        "07_Nic": "3",
        "08_Wang": "3",
    }

    scene_name = "carla_town04"

    root_path = Path("/path/to/dataset/selected_data")

    # Run this for copy file first. 
    # for driver, level in driver_list.items():
    #     source_path = Path(root_path, driver)
    #     copy_selected_seq_to_destination(source_path, root_path, level)

    
    for i in range(len(head_list)):
        name = f"head{head_list[i]}"
        head_file = f"carla_parking_manual_hm_gsp_Apr02_seq_head{head_list[i]}" 

        # Original copy function paths
        source_path = Path('/path/to/dataset/selected_data/', name)
        des_seq_path = Path("/home/USER/workspace/dataset/", head_file)

        destination_path = des_seq_path

        copy_seq_images(source_path, destination_path)
        
        # Run the division
        divide_train_val(destination_path, val_sample_ratio)
        print("Division completed!")
