import os
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_tar_file(tar_file_path, train_ratio, val_ratio, test_ratio):
    # Validate input
    if not os.path.exists(tar_file_path):
        print(f"Error: File '{tar_file_path}' does not exist.")
        return
    
    # Validate ratio sum
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        print("Error: The sum of train, validation, and test ratios must equal 1.0.")
        return
    
    # Extract .tar file
    file_name = os.path.splitext(os.path.basename(tar_file_path))[0]
    dataset_dir = os.path.join(os.getcwd(), 'dataset', file_name)
    data_dir = os.path.join(os.getcwd(), 'data', file_name)
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(path=dataset_dir)
    print(f"Extracted .tar file to {dataset_dir}")
    
    # Locate the CSV file
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV file found in extracted .tar archive at {dataset_dir}.")
        return
    
    csv_file_path = os.path.join(dataset_dir, csv_files[0])
    print(f"Found CSV file: {csv_file_path}")
    
    # Load and preprocess CSV file
    df = pd.read_csv(csv_file_path)
    
    # Replace NaN values with 0
    df.fillna(0, inplace=True)
    
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    if 'target' not in df.columns:
        print("Error: 'target' column not found in CSV file.")
        return
    
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    # Split into train, val, test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
    
    # Save splits to .npy files
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    print(f"Data splits saved to {data_dir}")

if __name__ == "__main__":
    import sys
    
    # Validate command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python src/data/preprocess_data.py <path_to_tar_file> <train_ratio> <val_ratio> <test_ratio>")
        sys.exit(1)
    
    tar_file_path = sys.argv[1]
    try:
        train_ratio = float(sys.argv[2])
        val_ratio = float(sys.argv[3])
        test_ratio = float(sys.argv[4])
    except ValueError:
        print("Error: Train, validation, and test ratios must be numeric values.")
        sys.exit(1)
    
    # Process the .tar file
    process_tar_file(tar_file_path, train_ratio, val_ratio, test_ratio)
