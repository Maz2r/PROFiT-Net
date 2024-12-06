import os
import sys

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.pytorch.cnn_model import CNNModel
from src.utils.target_labels import TargetLabels
import wandb

def predict_model(target_abbreviation, run_id):
    # Initialize WandB
    wandb.init(
        project="SePROFiT-Net",  # Replace with your WandB project name
        name=f"predict_1d_{target_abbreviation}_{run_id}",
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

    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        wandb.finish()
        return

    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_test, y_test = torch.tensor(X_test).float().unsqueeze(1), torch.tensor(y_test).float()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    # Fetch model weights from WandB
    artifact_name = f"{target_abbreviation}_{run_id}:latest"
    artifact = wandb.use_artifact(artifact_name, type='model')
    artifact_dir = artifact.download()

    # Load model
    checkpoint_path = os.path.join(artifact_dir, f"{target_abbreviation}_{run_id}_cp.pt")
    model = CNNModel(X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    # Make predictions
    predictions, answers, differences = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze().cpu().numpy()
            predictions.extend(preds)
            answers.extend(y_batch.numpy())
            differences.extend(np.abs(preds - y_batch.numpy()))

    # Save predictions and answers to a file
    output_file = os.path.join(os.getcwd(), f"predictions.txt")
    with open(output_file, "w") as f:
        f.write(f"{'Answer':<15}{'Prediction':<15}{'Difference':<15}\n")
        f.write(f"{'-'*45}\n")
        for answer, prediction, diff in zip(answers, predictions, differences):
            f.write(f"{answer:<15.4f}{prediction:<15.4f}{diff:<15.4f}\n")

    print(f"Predictions, answers, and differences saved to {output_file}")

    # Log predictions summary to WandB
    wandb.log({
        "mean_absolute_error": np.mean(np.abs(np.array(answers) - np.array(predictions))),
        "mean_squared_error": np.mean((np.array(answers) - np.array(predictions)) ** 2)
    })

    wandb.finish()


if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python src/predict/pytorch/predict.py <target_abbreviation> <run_id>")
        sys.exit(1)

    target_abbreviation = sys.argv[1]
    run_id = sys.argv[2]

    predict_model(target_abbreviation, run_id)
