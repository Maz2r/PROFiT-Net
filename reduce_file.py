import numpy as np
import os
import random

# Set paths
original_folder = "/home/maz2r/projects/PROFiT-Net/data/pbe_+u_band_gap_original/"  # Replace with the path to the folder containing original .npy files
reduced_folder = "/home/maz2r/projects/PROFiT-Net/data/pbe_+u_band_gap/"   # Replace with the path to the folder for saving reduced .npy files

# Create the reduced folder if it doesn't exist
os.makedirs(reduced_folder, exist_ok=True)

# Define the reduction factor (e.g., 0.5 for 50% reduction)
reduction_factor = 0.2

# List of files to process
files_to_reduce = ["X_train.npy", "X_val.npy", "X_test.npy", "y_train.npy", "y_val.npy", "y_test.npy"]

for file_name in files_to_reduce:
    # Load the file
    original_file_path = os.path.join(original_folder, file_name)
    data = np.load(original_file_path)
    
    # Determine the number of rows to keep
    total_rows = data.shape[0]
    reduced_rows = int(total_rows * reduction_factor)
    
    # Randomly select rows
    selected_indices = random.sample(range(total_rows), reduced_rows)
    reduced_data = data[selected_indices]
    
    # Save the reduced file
    reduced_file_path = os.path.join(reduced_folder, file_name)
    np.save(reduced_file_path, reduced_data)
    
    print(f"Reduced {file_name} from {total_rows} rows to {reduced_rows} rows and saved to {reduced_folder}")
