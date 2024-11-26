import sys
import os

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data.data_loader import DataLoader
from src.models.keras.cnn_model import CNNModel
from src.utils.callbacks import Callbacks
from src.utils.target_labels import TargetLabels
import tensorflow as tf

def train_model(target_abbreviation):
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
    
    # Load preprocessed data using DataLoader
    data_loader = DataLoader()
    X_train, X_val, y_train, y_val = data_loader.load_data(
        os.path.join(data_dir, 'X_train.npy'),
        os.path.join(data_dir, 'X_val.npy'),
        os.path.join(data_dir, 'y_train.npy'),
        os.path.join(data_dir, 'y_val.npy')
    )

    # Define input shape based on the training data
    input_shape = (X_train.shape[1], 1)

    # Build model
    model = CNNModel.build_model(input_shape)

    # Compile model
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['MeanAbsoluteError']
    )

    # Create checkpoint callback
    checkpoint_dir = os.path.join('callback', target_abbreviation)
    checkpoint_file = os.path.join(f'callback/{target_abbreviation}', f'{target_abbreviation}_cp.ckpt')
    callback = Callbacks.create_checkpoint_callback(checkpoint_file)

    # Train model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=10,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[callback]
    )

if __name__ == "__main__":
    import sys
    
    # Validate command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python src/train/keras/train.py <target_abbreviation>")
        sys.exit(1)
    
    target_abbreviation = sys.argv[1]
    train_model(target_abbreviation)
