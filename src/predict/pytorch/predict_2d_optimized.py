import os
import sys
import argparse

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.pytorch.cnn_model_2d_optuna import CNNModel_2D  # 변경된 임포트 경로
from src.utils.target_labels import TargetLabels
import wandb

def predict_model_2d(target_abbreviation, run_id):
    # Initialize WandB
    wandb.init(
        project="SePROFiT-Net",
        name=f"predict_2d_{target_abbreviation}_{run_id}",
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
    X_test = np.load(os.path.join(data_dir, 'X_test.npy')).reshape(-1, 1, 136, 136)
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    # Define config directly in code (Ensure this matches the training config)
    config = {
        'conv_params_list': [
            {'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'dropout': 0.08712642791973504, 'padding': 1, 'pooling': True, 'pool_type': 'max', 'pool_kernel_size': 2, 'pool_stride': 2}, 
            {'out_channels': 128, 'kernel_size': 5, 'stride': 1, 'dropout': 0.2637521839803184, 'padding': 2, 'pooling': True, 'pool_type': 'avg', 'pool_kernel_size': 2, 'pool_stride': 3}, 
            {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'dropout': 0.14799822825817954, 'padding': 1, 'pooling': False}, 
            {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'dropout': 0.04423604806894767, 'padding': 1, 'pooling': False}, 
            {'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'dropout': 0.2790605562309633, 'padding': 2, 'pooling': False}, 
            {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'dropout': 0.03979758734003716, 'padding': 2, 'pooling': False}
        ],
        # 추가적으로 필요한 설정이 있다면 여기 추가
    }

    # Fetch model artifact from WandB
    artifact_name = f'cnmd-phb-postech/SePROFiT-Net/{target_abbreviation}_{run_id}:latest'
    try:
        artifact = wandb.use_artifact(artifact_name, type='model')
    except wandb.errors.CommError as e:
        print(f"Error: {e}")
        wandb.finish()
        return
    except wandb.errors.Error as e:
        print(f"WandB Error: {e}")
        wandb.finish()
        return

    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")

    # Define checkpoint path based on config
    checkpoint_path = os.path.join(artifact_dir, f"{target_abbreviation}_{run_id}_cp.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist in the artifact.")
        wandb.finish()
        return

    # Initialize the model with config
    conv_params_list = config.get('conv_params_list', [])
    if not conv_params_list:
        print("Error: 'conv_params_list' is missing in the configuration.")
        wandb.finish()
        return

    model = CNNModel_2D(conv_params_list=conv_params_list).to(device)

    # Load model weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        wandb.finish()
        return

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
    output_file = os.path.join(os.getcwd(), f"predictions_2d_{run_id}.txt")
    with open(output_file, "w") as f:
        f.write(f"{'Answer':<15}{'Prediction':<15}{'Difference':<15}\n")
        f.write(f"{'-'*45}\n")
        for answer, prediction, diff in zip(answers, predictions, differences):
            f.write(f"{answer:<15.4f}{prediction:<15.4f}{diff:<15.4f}\n")

    print(f"Predictions, answers, and differences saved to {output_file}")

    # Log predictions summary to WandB
    mean_absolute_error = np.mean(np.abs(np.array(answers) - np.array(predictions)))
    mean_squared_error = np.mean((np.array(answers) - np.array(predictions)) ** 2)

    wandb.log({
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error
    })

    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Mean Squared Error: {mean_squared_error:.4f}")

    wandb.finish()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np  # numpy import 추가

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict using a 2D CNN model.")
    parser.add_argument("target_abbreviation", type=str, help="Target label abbreviation (e.g., exp_bg).")
    parser.add_argument("run_id", type=str, help="Run ID of the trained model.")

    args = parser.parse_args()

    predict_model_2d(args.target_abbreviation, args.run_id)
