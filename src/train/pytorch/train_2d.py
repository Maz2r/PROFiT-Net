import os
import sys
import argparse

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.pytorch.cnn_model_2d import CNNModel_2D
from src.utils.gpu_monitor import log_gpu_usage
from src.utils.target_labels import TargetLabels
from sklearn.metrics import mean_absolute_error
import wandb


def train_model(target_abbreviation, config):
    # Initialize WandB
    wandb.init(
        project="SePROFiT-Net",  # Replace with your WandB project name
        name=f"train_2d_{target_abbreviation}",
        entity='cnmd-phb-postech',
        config=config
    )

    # Generate a unique checkpoint name using WandB run ID
    run_id = wandb.run.id

    # Map abbreviation to full target name
    targets = TargetLabels.get_all_targets()
    if target_abbreviation not in targets:
        print(f"Error: Invalid target abbreviation '{target_abbreviation}'.")
        print(f"Valid abbreviations: {list(targets.keys())}")
        wandb.finish()
        return

    target_full_name = targets[target_abbreviation]
    data_dir = os.path.join(os.getcwd(), 'data', target_full_name)
    checkpoint_dir = os.path.join(f'callback_torch_2d/{target_abbreviation}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{target_abbreviation}_{run_id}_cp.pt')

    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        wandb.finish()
        return

    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy')).reshape(-1, 1, 136, 136)
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy')).reshape(-1, 1, 136, 136)
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config['batch_size'])

    model = CNNModel_2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()

    best_val_mae = float('inf')
    best_model_path = None

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, val_predictions = 0, []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())

        val_mae = mean_absolute_error(y_val.numpy(), np.array(val_predictions))

        # Save the best model locally
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), checkpoint_path)
            best_model_path = checkpoint_path
            print(f"Model saved to {checkpoint_path}")

        # Log metrics and GPU usage to WandB
        log_gpu_usage()  # Optional: Logs GPU usage to the console
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_mae': val_mae,
            'best_val_mae': best_val_mae,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

    # Log the best model weights as an artifact to WandB after training
    if best_model_path:
        artifact = wandb.Artifact(
            f'{target_abbreviation}_{run_id}',
            type='model',
            description=f'Best model weights for {target_abbreviation} (2D CNN)',
            metadata={'run_id': run_id}
        )
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    print(f"Training complete. Best Val MAE: {best_val_mae:.4f}")
    wandb.finish()


if __name__ == "__main__":
    # Basic configuration
    basic_config = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 500,
        'target_mae': 0.65,
        'target_mae_deviation': 0.03,
        'patience': 5,
    }
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a 2D CNN model.")
    parser.add_argument("target_abbreviation", type=str, help="Target label abbreviation (e.g., exp_bg).")
    parser.add_argument("--batch_size", type=int, default=basic_config['batch_size'], help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=basic_config['learning_rate'], help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=basic_config['epochs'], help="Number of epochs.")
    parser.add_argument("--target_mae", type=float, default=basic_config['target_mae'], help="Target MAE for early stopping.")
    parser.add_argument("--target_mae_deviation", type=float, default=basic_config['target_mae_deviation'], help="Target MAE deviation.")
    parser.add_argument("--patience", type=int, default=basic_config['patience'], help="Patience for early stopping.")
    parser.add_argument("--wandb_api", type=str, default="src/wandb_apikey.txt", help="Path to WandB API key file.")

    args = parser.parse_args()

    # Initialize WandB
    with open(args.wandb_api, 'r') as f:
        wandb_api_key = f.read().strip()
    
    wandb.login(key=wandb_api_key)

    # Update configuration with command-line arguments
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'target_mae': args.target_mae,
        'target_mae_deviation': args.target_mae_deviation,
        'patience': args.patience,
    }

    train_model(args.target_abbreviation, config)
