import pandas as pd
import os

# Define a function to find non-numeric values in the dataframe
def find_inadequate_values(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Drop the "id" column if it exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    # List to store column, row, and inadequate value information
    inadequate_values = []

    # Iterate through all columns
    for col in df.columns:
        # Check if the column contains numeric values
        for row_idx, value in enumerate(df[col]):
            try:
                # Try converting the value to float
                float(value)
            except ValueError:
                # If it fails, the value is not numeric
                inadequate_values.append((col, row_idx, value))
    
    return inadequate_values

# Define a function to write the results to a .txt file
def write_inadequate_values_to_txt(file_path, inadequate_values):
    # Generate the output .txt file name
    output_file = os.path.splitext(file_path)[0] + "_inadequate_values.txt"
    
    # Write to the .txt file
    with open(output_file, "w") as f:
        if inadequate_values:
            f.write("Inadequate (non-numeric) values found:\n")
            for col, row, value in inadequate_values:
                f.write(f"Column: {col}, Row: {row}, Value: {value}\n")
        else:
            f.write("No inadequate (non-numeric) values found.\n")

# Path to your CSV file
file_path = './dataset/pbe_+u_band_gap/pbe_+u_band_gap.csv'  # Replace with your actual file path

# Find inadequate values
inadequate_values = find_inadequate_values(file_path)

# Write the results to a .txt file
write_inadequate_values_to_txt(file_path, inadequate_values)

print(f"Results have been written to {os.path.splitext(file_path)[0]}_inadequate_values.txt")
