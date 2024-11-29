import os
import sys

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import torch
from src.models.pytorch.cnn_model import CNNModel
from src.utils.target_labels import TargetLabels
from sklearn.metrics import mean_absolute_error
import wandb


def predict_model(target_abbreviation, run_id):
    # Initialize WandB
    wandb.init(
        project="SePROFiT-Net",  # Replace with your WandB project name
        name=f"predict_{target_abbreviation}_{run_id}",
        entity='cnmd-phb-postech'
    )

    # Map abbreviation to full target name
    targets = TargetLabels.get_all_targets()
    if target_abbreviation not in targets:
        print(f"Error: Invalid target abbreviation '{target_abbreviation}'.")
        print(f"Valid abbreviations: {list(targets.keys())}")
        wandb.finish()
        return

    target_full_name = targets[target_abbreviation]
    data_dir = os.path.join(os.getcwd(), 'data', target_full_name)
    local_weights_dir = os.path.join(f'fetched_weights/{target_abbreviation}')
    os.makedirs(local_weights_dir, exist_ok=True)
    weights_path = os.path.join(local_weights_dir, f'{target_abbreviation}_{run_id}_cp.pt')

    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        wandb.finish()
        return

    # Fetch model weights from WandB
    artifact_name = f"{target_abbreviation}_{run_id}:latest"
    artifact = wandb.use_artifact(artifact_name, type='model')
    artifact_dir = artifact.download(local_weights_dir)
    weights_path = os.path.join(artifact_dir, f"{target_abbreviation}_{run_id}_cp.pt")

    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' does not exist.")
        wandb.finish()
        return

    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_test, y_test = torch.tensor(X_test).float().unsqueeze(1).to(device), torch.tensor(y_test).float().to(device)

    # Load model
    model = CNNModel(X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(weights_path, weights_only = True))
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(X_test).squeeze().cpu().numpy()

    # Calculate metrics
    mae = mean_absolute_error(y_test.cpu().numpy(), predictions)
    mse = np.mean((y_test.cpu().numpy() - predictions) ** 2)

    # Log results to WandB
    wandb.log({
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
    })

    # Print detailed prediction summary
    print(f"Prediction Summary for {target_abbreviation} (Run ID: {run_id}):")
    print(f"- Mean Absolute Error (MAE): {mae:.4f}")
    print(f"- Mean Squared Error (MSE): {mse:.4f}")
    print(f"- Predictions (Sample): {predictions[:10]}")  # Print first 10 predictions

    # Finish WandB session
    wandb.finish()


if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python src/predict/pytorch/predict.py <target_abbreviation> <run_id>")
        sys.exit(1)

    target_abbreviation = sys.argv[1]
    run_id = sys.argv[2]

    predict_model(target_abbreviation, run_id)
