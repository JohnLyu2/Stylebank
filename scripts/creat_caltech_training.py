import os
import shutil
import random

# Define source and destination directories
source_dir = '/Users/zhengyanglumacmini/Desktop/Projects/Stylebank/101_ObjectCategories'
dest_dir = '/Users/zhengyanglumacmini/Desktop/Projects/Stylebank/caltech_101_training'

# Ensure the destination directory exists
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through each subfolder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # Check if it's indeed a folder
    if os.path.isdir(folder_path):
        # List all files in the subfolder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # Randomly select 10 images, or all if there are fewer than 10
        selected_files = random.sample(files, min(10, len(files)))
        
        # Destination subfolder path
        dest_subfolder_path = os.path.join(dest_dir, folder_name)
        
        # Ensure the destination subfolder exists
        if not os.path.exists(dest_subfolder_path):
            os.makedirs(dest_subfolder_path)
        
        # Move each selected file to the destination subfolder
        for file_name in selected_files:
            source_file_path = os.path.join(folder_path, file_name)
            dest_file_path = os.path.join(dest_subfolder_path, file_name)
            shutil.move(source_file_path, dest_file_path)

print("Training set created successfully.")
