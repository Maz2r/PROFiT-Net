import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import wandb
import time
import pynvml  # GPU usage logging
import optuna
from optuna.trial import TrialState

target_prop = "exp_band_gap"

# GPU monitoring initialization
pynvml.nvmlInit()


# Wandb setup
api_key = 'bb88a081ad598db2b81c11dc2621ca6011649c55'
wandb.login(key=api_key)
wandb.init(
    project='SePROFiT-Net',
    name='exp_band_gap_2dcnn_optuna_optimization',
    entity='cnmd-phb-postech',
    config={
        'batch_size': 128,
        'learning_rate': 0.001,
        # 'epochs': 500,
        'target_mae': 0.65,
        'target_mae_deviation': 0.03,
        'patience': 5,
    }
)
config = wandb.config


# Data loading
X_train = np.load(f'{target_prop}/X_train.npy').reshape(-1, 1, 136, 136)
X_val = np.load(f'{target_prop}/X_val.npy').reshape(-1, 1, 136, 136)
y_train = np.load(f'{target_prop}/y_train.npy')
y_val = np.load(f'{target_prop}/y_val.npy')


# Convert data to tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).float()
y_val = torch.from_numpy(y_val).float()


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Optuna objective function
def objective(trial):
    n_conv_layers = 5
    conv_channels = []
    kernel_sizes = []
    strides = []
    pooling_methods = []
    pooling_kernel_sizes = []
    pooling_strides = []
    dropout_rates = []

    current_height, current_width = 136, 136

    # First two convolutional layers with fixed kernel sizes and pooling
    # First convolutional layer
    out_channels = 32
    kernel_size = trial.suggest_int(f'conv_1_kernel_size', 3, 32)
    stride = trial.suggest_int(f'conv_0_stride', 1, 2)
    dropout_rate = trial.suggest_float(f'dropout_rate_0', 0.0, 0.1)
    pooling_method = trial.suggest_categorical(f'pooling_method_0', ['MaxPool2d', 'AvgPool2d'])
    pooling_kernel_size = trial.suggest_int(f'pooling_kernel_size_0', 2, min(8, current_height, current_width))
    pooling_stride = trial.suggest_int(f'pooling_stride_0', 1, 3)

    conv_channels.append(out_channels)
    kernel_sizes.append(kernel_size)
    strides.append(stride)
    dropout_rates.append(dropout_rate)
    pooling_methods.append(pooling_method)
    pooling_kernel_sizes.append(pooling_kernel_size)
    pooling_strides.append(pooling_stride)

    current_height = (current_height - kernel_size) // stride + 1
    current_width = (current_width - kernel_size) // stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    current_height = (current_height - pooling_kernel_size) // pooling_stride + 1
    current_width = (current_width - pooling_kernel_size) // pooling_stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    # Second convolutional layer
    out_channels = 64
    kernel_size = trial.suggest_int(f'conv_1_kernel_size', 3, 32)
    # kernel_size = 64  # Fixed kernel size
    stride = trial.suggest_int(f'conv_1_stride', 1, 3)
    dropout_rate = trial.suggest_float(f'dropout_rate_1', 0.0, 0.1)
    pooling_method = trial.suggest_categorical(f'pooling_method_1', ['MaxPool2d', 'AvgPool2d'])
    pooling_kernel_size = trial.suggest_int(f'pooling_kernel_size_1', 2, min(8, current_height, current_width))
    pooling_stride = trial.suggest_int(f'pooling_stride_1', 1, 3)

    conv_channels.append(out_channels)
    kernel_sizes.append(kernel_size)
    strides.append(stride)
    dropout_rates.append(dropout_rate)
    pooling_methods.append(pooling_method)
    pooling_kernel_sizes.append(pooling_kernel_size)
    pooling_strides.append(pooling_stride)

    current_height = (current_height - kernel_size) // stride + 1
    current_width = (current_width - kernel_size) // stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    current_height = (current_height - pooling_kernel_size) // pooling_stride + 1
    current_width = (current_width - pooling_kernel_size) // pooling_stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    # Third convolutional layer
    out_channels = trial.suggest_int(f'conv_2_out_channels', 2, conv_channels[-1])
    kernel_size = trial.suggest_int(f'conv_2_kernel_size', 8, 20)
    stride = trial.suggest_int(f'conv_2_stride', 1, 3)
    dropout_rate = trial.suggest_float(f'dropout_rate_2', 0.0, 0.1)

    conv_channels.append(out_channels)
    kernel_sizes.append(kernel_size)
    strides.append(stride)
    dropout_rates.append(dropout_rate)

    current_height = (current_height - kernel_size) // stride + 1
    current_width = (current_width - kernel_size) // stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    # Fourth convolutional layer
    out_channels = trial.suggest_int(f'conv_3_out_channels', 2, conv_channels[-1])
    previous_kernel_size = kernel_sizes[-1]
    kernel_size = trial.suggest_int(f'conv_3_kernel_size', 3, previous_kernel_size - 1)
    stride = trial.suggest_int(f'conv_3_stride', 1, 3)
    dropout_rate = trial.suggest_float(f'dropout_rate_3', 0.0, 0.1)

    conv_channels.append(out_channels)
    kernel_sizes.append(kernel_size)
    strides.append(stride)
    dropout_rates.append(dropout_rate)

    current_height = (current_height - kernel_size) // stride + 1
    current_width = (current_width - kernel_size) // stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    # Fifth convolutional layer
    out_channels = trial.suggest_int(f'conv_4_out_channels', 2, conv_channels[-1])
    kernel_size = 1
    stride = trial.suggest_int(f'conv_4_stride', 1, 3)
    dropout_rate = trial.suggest_float(f'dropout_rate_4', 0.0, 0.1)

    conv_channels.append(out_channels)
    kernel_sizes.append(kernel_size)
    strides.append(stride)
    dropout_rates.append(dropout_rate)

    current_height = (current_height - kernel_size) // stride + 1
    current_width = (current_width - kernel_size) // stride + 1
    if current_height <= 0 or current_width <= 0:
        raise optuna.exceptions.TrialPruned()

    # Activation and optimizer
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU'])
    activation = getattr(nn, activation_name)()
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    # Model definition
    class My2DModel(nn.Module):
        def __init__(self):
            super(My2DModel, self).__init__()

            layers = []
            in_channels = 1

            for i in range(n_conv_layers):
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[i], kernel_size=kernel_sizes[i], stride=strides[i]))
                layers.append(activation)
                if i < 2:  # Apply pooling only to the first two layers
                    layers.append(getattr(nn, pooling_methods[i])(kernel_size=pooling_kernel_sizes[i], stride=pooling_strides[i]))
                if dropout_rates[i] > 0:
                    layers.append(nn.Dropout(dropout_rates[i]))

                in_channels = conv_channels[i]

            self.conv_layers = nn.Sequential(*layers)
            self._to_linear = in_channels * current_height * current_width
            self.fc_layers = nn.Sequential(
                nn.Linear(self._to_linear, 1024),
                activation,
                nn.Linear(1024, 256),
                activation,
                nn.Linear(256, 1)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

    model = My2DModel().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Training loop
    best_val_mae = float('inf')
    for epoch in range(50):  # Reduced epochs for quick optimization
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.sum(torch.abs(outputs - targets)).item()

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_mae += torch.sum(torch.abs(outputs - targets)).item()

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_mae < best_val_mae:
            best_val_mae = val_mae

    return best_val_mae


# Optuna study creation and optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters:')
for key, value in study.best_params.items():
    print(f'{key}: {value}')
print(f'Best validation MAE: {study.best_value}')

wandb.finish()
pynvml.nvmlShutdown()
