import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import wandb
import pandas as pd
import time
import optuna

# Add project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from src.models.pytorch.cnn_model_2d_optuna import CNNModel_2D
from src.utils.gpu_monitor import log_gpu_usage
from src.utils.target_labels import TargetLabels

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for 2D CNN model with independent channels per layer.")
    parser.add_argument("target_abbreviation", type=str, help="Target label abbreviation (e.g., exp_bg).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs per trial.")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials for optimization.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per trial.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training.")
    parser.add_argument("--wandb_api", type=str, default="src/wandb_apikey.txt", help="Path to WandB API key file.")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load WandB API Key and login
    with open(args.wandb_api, 'r') as f:
        wandb_api_key = f.read().strip()
    wandb.login(key=wandb_api_key)

    # Load data
    target_abbreviation = args.target_abbreviation
    targets = TargetLabels.get_all_targets()
    if target_abbreviation not in targets:
        print(f"Error: Invalid target abbreviation '{target_abbreviation}'.")
        print(f"Valid abbreviations: {list(targets.keys())}")
        return

    target_full_name = targets[target_abbreviation]
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data', target_full_name)

    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    X_train = np.load(os.path.join(data_dir, 'X_train.npy')).reshape(-1, 1, 136, 136)
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy')).reshape(-1, 1, 136, 136)
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)

    # CSV file to save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_dir = os.path.join(script_dir, '../../../train_history')
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f'optuna_2DCNN_{args.target_abbreviation}_5layers_independent_channels.csv')

    # Create study
    study = optuna.create_study(direction='minimize', study_name=f"optuna_study_{args.target_abbreviation}_5layers_independent_channels")

    # *** Initial heuristic hyperparameters ***
    initial_heuristic = {
        # Define initial values for hyperparameters to be tuned by BO
        'c1': 32,
        'c2': 64,
        'c3': 32,
        'c4': 16,
        'c5': 8,
        'kernel_size_0': 5,
        'stride_0': 1,
        'dropout_0': 0.02,
        'padding_0': 1,
        'pool_type_0': 'max',
        'pool_kernel_size_0': 2,
        # 'pool_stride_0': 2,
        'kernel_size_1': 5,
        'stride_1': 1,
        'dropout_1': 0.03,
        'padding_1': 1,
        'pool_type_1': 'avg',
        'pool_kernel_size_1': 2,
        'pool_stride_1': 2,
        'kernel_size_2': 3,
        'stride_2': 1,
        'dropout_2': 0.01,
        'padding_2': 1,
        'kernel_size_3': 3,
        'stride_3': 1,
        'dropout_3': 0.01,
        'padding_3': 0,
        'kernel_size_4': 3,
        'stride_4': 1,
        'dropout_4': 0.01,
        'padding_4': 0,
    }

    # Enqueue initial hyperparameters
    study.enqueue_trial(initial_heuristic)

    # Define objective function
    def objective(trial):
        repeat_mae_list = []

        for repeat_idx in range(args.repeats):
            # Initialize WandB for each repeat
            wandb.init(
                project="SePROFiT-Net",
                name=f"optuna_trial_{args.target_abbreviation}_trial{trial.number}_repeat{repeat_idx}",
                entity='cnmd-phb-postech',
                config={
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'epochs': args.epochs,
                    'trial_number': trial.number,
                    'repeat_index': repeat_idx
                }
            )

            # Hyperparameters
            learning_rate = args.learning_rate
            epochs = args.epochs

            # Number of convolutional layers (fixed to 5)
            num_conv_layers = 5

            conv_params_list = []

            # Define output channels for each layer independently
            c1 = trial.suggest_categorical('c1', [8, 16, 32, 64, 128])
            c2 = trial.suggest_categorical('c2', [8, 16, 32, 64, 128, 256])
            c3 = trial.suggest_categorical('c3', [8, 32, 64, 128, 256, 512])
            c4 = trial.suggest_categorical('c4', [8, 16, 32, 64, 128, 256])
            c5 = trial.suggest_categorical('c5', [8, 16, 32, 64, 128])

            output_channels_list = [c1, c2, c3, c4, c5]

            for i in range(num_conv_layers):
                conv_params = {}

                conv_params['out_channels'] = output_channels_list[i]

                # Kernel size
                conv_params['kernel_size'] = trial.suggest_categorical(f'kernel_size_{i}', [3, 5])

                # Stride
                conv_params['stride'] = trial.suggest_int(f'stride_{i}', 1, 2)

                # Dropout probability
                conv_params['dropout'] = trial.suggest_float(f'dropout_{i}', 0.0, 0.3)

                # Padding
                conv_params['padding'] = trial.suggest_int(f'padding_{i}', 0, 2)

                # For the first two layers, include pooling
                if i < 2:
                    conv_params['pooling'] = True
                    conv_params['pool_type'] = trial.suggest_categorical(f'pool_type_{i}', ['max', 'avg'])
                    conv_params['pool_kernel_size'] = trial.suggest_int(f'pool_kernel_size_{i}', 2, 3)
                    conv_params['pool_stride'] = trial.suggest_int(f'pool_stride_{i}', 2, 3)
                else:
                    conv_params['pooling'] = False

                conv_params_list.append(conv_params)

            # Build model
            try:
                model = CNNModel_2D(conv_params_list).to(device)
            except Exception as e:
                print(f"Error in model creation: {e}")
                wandb.finish()
                return float('inf')

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            params = sum(p.numel() for p in model.parameters())  # Number of parameters

            best_val_mae = float('inf')

            for epoch in range(epochs):
                epoch_start_time = time.time()

                model.train()
                train_loss = 0
                train_mae = 0

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    predictions = model(X_batch).squeeze()
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                    train_mae += torch.sum(torch.abs(predictions - y_batch)).item()

                total_train_samples = len(train_loader.dataset)
                train_loss = train_loss / total_train_samples
                train_mae = train_mae / total_train_samples

                # Validation
                model.eval()
                val_loss = 0
                val_mae = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        predictions = model(X_batch).squeeze()
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                        val_mae += torch.sum(torch.abs(predictions - y_batch)).item()

                total_val_samples = len(val_loader.dataset)
                val_loss = val_loss / total_val_samples
                val_mae = val_mae / total_val_samples

                # Log to WandB
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_mae': train_mae,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'best_val_mae': best_val_mae,
                    'learning_rate': learning_rate,
                    'epoch_time': time.time() - epoch_start_time
                })

                # Update best validation MAE
                if val_mae < best_val_mae:
                    best_val_mae = val_mae

            # Append the MAE of this repeat
            repeat_mae_list.append(best_val_mae)

            # Finish WandB run
            wandb.finish()

        # Calculate the mean MAE over the repeats
        mean_val_mae = np.mean(repeat_mae_list)

        # Save trial results
        result = {
            'trial_number': trial.number,
            'mean_val_mae': mean_val_mae,
            'individual_maes': repeat_mae_list,
            'params': params,
            'conv_params_list': conv_params_list
        }

        df_result = pd.DataFrame([result])
        if not os.path.exists(history_file):
            df_result.to_csv(history_file, index=False)
        else:
            df_result.to_csv(history_file, mode='a', header=False, index=False)

        return mean_val_mae  # Objective value to minimize

    # Optimize
    study.optimize(objective, n_trials=args.trials)

    # Save study results
    study.trials_dataframe().to_csv(os.path.join(history_dir, f'study_2DCNN_{args.target_abbreviation}_5layers_independent_channels.csv'), index=False)

if __name__ == "__main__":
    main()
