import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
import pdb
import random

def find_camera_folders(root_path, camera_folders):
    """
    Recursively find all paths containing the specified camera folders
    Args:
        root_path: Root directory to start searching from
        camera_folders: List of camera folder names to search for
    Returns:
        Dictionary mapping camera folder names to lists of paths
    """
    camera_paths = {folder: [] for folder in camera_folders}
    
    def search_recursive(current_path):
        try:
            # List all items in current directory
            for item in current_path.iterdir():
                if item.is_dir():
                    # Check if this directory is one of our target camera folders
                    if item.name in camera_folders:
                        camera_paths[item.name].append(item)
                    else:
                        # If not, continue searching in this directory
                        search_recursive(item)
        except PermissionError:
            print(f"Permission denied: {current_path}")
        except Exception as e:
            print(f"Error accessing {current_path}: {str(e)}")
    
    # Start recursive search from root path
    print(f"Starting search from: {root_path}")
    search_recursive(Path(root_path))
    
    return camera_paths

def calculate_mean_std(dataset_path, camera_folders, batch_size=32, sampling_interval=10):
    """
    Calculate mean and std of the dataset
    Args:
        dataset_path: Path to image directory
        batch_size: Batch size for processing
        sampling_interval: Interval for sampling images (process every Nth image)
    Returns:
        mean and std values for each channel
    """
    # Check if directory exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    # Convert images to tensors without normalization
    transform = transforms.Compose([
        transforms.ToTensor()  # Scales images to [0, 1]
    ])
    
    # Initialize variables for mean and std calculation
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    # Define camera folders to search for
    # camera_folders = ["rgb_front", "rgb_left", "rgb_right", "rgb_rear"]
    
    
    # Find all camera folders in the dataset
    print("Searching for camera folders...")
    camera_paths = find_camera_folders(dataset_path, camera_folders)
    
    print(f"camera_paths: {camera_paths}")
    
    # Get all image files from specified camera folders
    image_files = []
    error_count = 0
    total_files = 0
    
    for folder, paths in camera_paths.items():
        print(f"\nFound {len(paths)} directories for {folder}:")
        for path in paths:
            png_files = list(path.glob("*.png"))
            total_files += len(png_files)
            
            # Use random sampling instead of fixed interval sampling
            if sampling_interval > 1:
                sampled_files = random.sample(png_files, len(png_files) // sampling_interval)
            else:
                sampled_files = png_files
                
            image_files.extend(sampled_files)
            print(f"  {path}: {len(png_files)} images, sampled {len(sampled_files)} images")
    
    print(f"\nTotal number of images found: {len(image_files)} (after sampling)")
    print(f"Total files in dataset: {total_files}")
    print(f"Sampling rate: {len(image_files)/total_files:.2%}")
    
    if len(image_files) == 0:
        raise ValueError(f"No PNG images found in any of the camera folders: {camera_folders}")
    
    # First pass: calculate mean
    print("\nCalculating mean...")
    processed_count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            mean += img_tensor.mean(dim=[1, 2])
            total_images += 1
            processed_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    if total_images == 0:
        raise ValueError("No valid images were processed")
    
    mean /= total_images
    
    # Second pass: calculate std
    print("\nCalculating std...")
    processed_count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            std += ((img_tensor - mean.reshape(-1, 1, 1)) ** 2).mean(dim=[1, 2])
            processed_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    std = torch.sqrt(std / total_images)
    
    # Print final statistics
    print(f"\nProcessing completed with {error_count} errors")
    print(f"Successfully processed {processed_count} images")
    print(f"Error rate: {error_count/len(image_files):.2%}")
    
    return mean.numpy(), std.numpy()

if __name__ == "__main__":
    # Specify dataset path
    # dataset_path = "/path/to/dataset/total_sorted_apr20/"
    # dataset_path = "/path/to/dataset/total_sorted_apr21/"
    dataset_path = '/workspace/dataset/e2e_parking/total_sorted_apr29/train_Town04_Opt_sorted_filtered'
    # dataset_path = "/path/to/dataset/merge/"
    # Use tuples instead of lists for camera folders
    camera_folders = [
        ("rgb_front",),
        ("rgb_left",),
        ("rgb_right",),
        ("rgb_rear",),
        ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")
    ]
    
    for camera_folder in camera_folders:
        try:
            mean, std = calculate_mean_std(dataset_path, camera_folder)
            
            print("\nDataset statistics:")
            print(f"Camera folder: {camera_folder}")
            print(f"Mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
            print(f"Std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
            print(f"--------------------------------")
            # write into the log file
            with open("normalize_sorted_apr21.log", "a") as f:
                f.write(f"Camera folder: {camera_folder}\n")
                f.write(f"Mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]\n")
                f.write(f"Std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]\n")
                f.write(f"--------------------------------\n")  
        except Exception as e:
            print(f"Error: {str(e)}")
    
