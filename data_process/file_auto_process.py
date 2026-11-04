import shutil
from pathlib import Path
import pdb

val_sample_ratio = 0.1  # Adjust this number by requirements

def copy_seq_images(source_path, destination_path):
    """
    复制序列图像数据从源路径到目标路径，并按特定格式重命名。

    参数:
        source_path (Path): 源数据路径，期望包含多个子文件夹，每个子文件夹下有'seq'目录
        destination_path (Path): 目标路径，将在此创建'images'和'head1'子目录

    功能描述:
    1. 遍历source_path下的所有文件夹
    2. 在每个文件夹的'seq'目录下查找任务文件夹(task_X)
    3. 对每个任务文件夹:
       - 从'inputs'子目录复制PNG文件到destination_path/images/
       - 从'gt'子目录复制PNG文件到destination_path/head1/
    4. 复制时会重命名文件为: {scene_name}_{name}_{folder_name}_{task_name}_{original_filename}

    注意:
    - 使用全局变量scene_name和name作为文件名前缀
    - 自动创建目标目录结构
    - 保留原始文件的元数据(使用shutil.copy2)
    """
    
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

if __name__ == "__main__":

    name = "head2" # TODO: use it as the head group
    scene_name = "carla_town04"
    
    # Original copy function paths
    source_path = Path('/path/to/dataset/generated/', name)
    des_seq_path = Path("/path/to/dataset/output_seq")

    destination_path = des_seq_path
    copy_seq_images(source_path, destination_path)
    
    # Run the division
    divide_train_val(destination_path, val_sample_ratio)
    print("Division completed!")
