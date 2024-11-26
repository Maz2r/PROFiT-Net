import os
import sys

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.models.keras.cnn_model import CNNModel
from src.utils.target_labels import TargetLabels
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_model(target_abbreviation):
    # Map abbreviation to the full target name
    targets = TargetLabels.get_all_targets()
    if target_abbreviation not in targets:
        print(f"Error: Invalid target abbreviation '{target_abbreviation}'.")
        print(f"Valid abbreviations: {list(targets.keys())}")
        return

    target_full_name = targets[target_abbreviation]
    data_dir = os.path.join(os.getcwd(), 'data', target_full_name)
    
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return
    
    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Define input shape based on the test data
    input_shape = (X_test.shape[1], 1)

    # Build model
    model = CNNModel.build_model(input_shape)

    # Load weights
    checkpoint_dir = os.path.join('callback', target_abbreviation)
    checkpoint_file = os.path.join(checkpoint_dir, f'{target_abbreviation}_cp.ckpt')
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist.")
        return
    if not os.path.exists(checkpoint_file + ".index"):
        print(f"Error: Checkpoint file '{checkpoint_file}' does not exist.")
        return

    status = model.load_weights(checkpoint_file)
    status.expect_partial()

    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate and display statistics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\nModel Performance Statistics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python src/predict/keras/predict.py <target_abbreviation>")
        sys.exit(1)
    
    target_abbreviation = sys.argv[1]
    predict_model(target_abbreviation)
