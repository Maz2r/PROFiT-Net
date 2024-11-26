import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import wandb
import time
import pynvml  # GPU usage logging

target_prop = "exp_band_gap"

# NVML Initialization for GPU monitoring
pynvml.nvmlInit()

# Environment variable for API key
api_key = 'bb88a081ad598db2b81c11dc2621ca6011649c55'  # Better to use environment variables for security
if api_key is None:
    raise ValueError("API key is required")

# Log in to wandb using API key
wandb.login(key=api_key)

# Initialize wandb with project and entity
wandb.init(
    project='SePROFiT-Net',
    name='exp_band_gap_2dcnn',
    entity='cnmd-phb-postech',
    config={
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 500,
        'target_mae': 0.62,
        'target_mae_deviation': 0.03,
        'patience': 5,
    }
)
config = wandb.config

# Start tracking execution time
start_time = time.time()

# Load data
# Data loading
X_train = np.load(f'{target_prop}/X_train.npy').reshape(-1, 1, 136, 136)
X_val = np.load(f'{target_prop}/X_val.npy').reshape(-1, 1, 136, 136)
y_train = np.load(f'{target_prop}/y_train.npy')
y_val = np.load(f'{target_prop}/y_val.npy')

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the model
class My2DModel(nn.Module):
    def __init__(self):
        super(My2DModel, self).__init__()

        # 2D Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.02),

            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.03),

            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.05),

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.03),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.02),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(0.01),
        )

        # Calculate output size after convolutional layers
        self._to_linear = None
        self._get_conv_output()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _get_conv_output(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 136, 136)
            output = self.conv_layers(sample_input)
            self._to_linear = output.view(1, -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Instantiate the model
model = My2DModel()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Early stopping variables
target_mae = config.target_mae
target_mae_deviation = config.target_mae_deviation
patience = config.patience
val_mae_history = []
training_stopped = False

# Track best validation MAE
best_val_mae = float('inf')
checkpoint_path = 'callback/cp.pt'
os.makedirs('callback', exist_ok=True)

# Learning rate reduction and additional epochs
lr_reduced = False
additional_epochs = 10

# Training loop
for epoch in range(config.epochs):
    epoch_start_time = time.time()
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

    # Validation step
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

    # Track validation MAE
    val_mae_history.append(val_mae)

    # Save best model
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Validation MAE improved, saving model to {checkpoint_path}')
        wandb.save(checkpoint_path)

    # Early stopping
    if len(val_mae_history) >= patience:
        recent_mae = val_mae_history[-patience:]
        mae_deviation = max(recent_mae) - min(recent_mae)
        if val_mae <= target_mae and mae_deviation <= target_mae_deviation:
            if not lr_reduced:
                print(f'Validation MAE stabilized within {target_mae_deviation} for the last {patience} epochs.')
                print('Reducing learning rate by 1/10.')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f'New learning rate: {optimizer.param_groups[0]["lr"]}')
                lr_reduced = True
                additional_epochs = 10
            elif additional_epochs > 0:
                print(f'Training for {additional_epochs} additional epochs.')
                additional_epochs -= 1
                if additional_epochs == 0:
                    print('Additional epochs completed, stopping training.')
                    training_stopped = True

    if training_stopped:
        break

    # GPU usage logging
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    # Epoch duration
    epoch_duration = time.time() - epoch_start_time

    # Log metrics to wandb
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_mae': train_mae,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'gpu_memory_used': mem_info.used / (1024 ** 2),
        'gpu_utilization': gpu_util.gpu,
        'epoch_duration': epoch_duration,
    })

    print(f'Epoch {epoch+1}/{config.epochs}, '
          f'train loss: {train_loss:.4f}, train MAE: {train_mae:.4f}, '
          f'val loss: {val_loss:.4f}, val MAE: {val_mae:.4f}, '
          f'learning rate: {optimizer.param_groups[0]["lr"]:.6f}, '
          f'GPU memory usage: {mem_info.used / (1024 ** 2):.2f} MB, '
          f'GPU Utilization: {gpu_util.gpu}%, '
          f'Epoch Duration: {epoch_duration:.2f} sec')

# Total training time
total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} sec")
wandb.log({'total_training_time': total_time})

# Finish wandb run
wandb.finish()

# Shutdown NVML
pynvml.nvmlShutdown()
