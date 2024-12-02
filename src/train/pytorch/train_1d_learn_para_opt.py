import os
import sys
import argparse

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.pytorch.cnn_model import CNNModel
from src.utils.gpu_monitor import log_gpu_usage
from src.utils.target_labels import TargetLabels
from sklearn.metrics import mean_absolute_error
import wandb
import pandas as pd
import time  # 시간 측정을 위해 추가

def train_model(target_abbreviation, config):
    # Initialize WandB
    wandb.init(
        project="SePROFiT-Net",  # Replace with your WandB project name
        name=f"train_{target_abbreviation}_{config['num']}",
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
    checkpoint_dir = os.path.join(f'callback_torch/{target_abbreviation}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{target_abbreviation}_{run_id}_cp.pt')

    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        wandb.finish()
        return

    # Load data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train).float().unsqueeze(1), torch.tensor(y_train).float()
    X_val, y_val = torch.tensor(X_val).float().unsqueeze(1), torch.tensor(y_val).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config['batch_size'])

    model = CNNModel(X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()

    params = sum(p.numel() for p in model.parameters())  # 모델의 파라미터 수 계산

    best_val_mae = float('inf')
    best_model_path = None

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Epoch 시작 시간 측정

        # Training
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
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.sum(torch.abs(predictions - y_batch)).item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        total_val_samples = len(val_loader.dataset)
        val_loss = val_loss / total_val_samples
        val_mae = val_mae / total_val_samples

        # Save the best model locally
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), checkpoint_path)
            best_model_path = checkpoint_path
            print(f"Model saved to {checkpoint_path}")

        # Measure epoch time
        epoch_time = time.time() - epoch_start_time

        # Log metrics and GPU usage to WandB
        log_gpu_usage()  # Optional: Logs GPU usage to the console
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'best_val_mae': best_val_mae,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, "
              f"Time: {epoch_time:.2f}s")

    # 최종 결과 저장
    final_train_loss = train_loss
    final_train_mae = train_mae
    final_val_loss = val_loss
    final_val_mae = val_mae

    # Log the best model weights as an artifact to WandB after training
    if best_model_path:
        artifact = wandb.Artifact(
            f'{target_abbreviation}_{run_id}',
            type='model',
            description=f'Best model weights for {target_abbreviation}',
            metadata={'run_id': run_id}
        )
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    print(f"Training complete. Best Val MAE: {best_val_mae:.4f}")
    wandb.finish()

    return final_train_loss, final_train_mae, final_val_loss, final_val_mae, best_val_mae, params

if __name__ == "__main__":
    import pandas as pd
    import numpy as np  # numpy import 추가

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a CNN model.")
    parser.add_argument("target_abbreviation", type=str, help="Target label abbreviation (e.g., exp_bg).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--target_mae", type=float, default=0.65, help="Target MAE for early stopping.")
    parser.add_argument("--target_mae_deviation", type=float, default=0.03, help="Target MAE deviation.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--wandb_api", type=str, default="src/wandb_apikey.txt", help="Path to WandB API key file.")

    args = parser.parse_args()

    # CSV 파일 로드 또는 생성
    history_dir = 'train_history'
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f'1DCNN_{args.target_abbreviation}.csv')

    columns = ["num.", "train_loss", "train_mae", "val_loss", "val_mae", "best_val_mae",
               "batch_size", "learning_rate", "epoch", "end_epoch", "target_mae",
               "target_mae_deviation", "patience", "params"]

    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        # 마지막 행의 val_mae가 NaN인지 확인
        if np.isnan(df.iloc[-1]['val_mae']):
            df = df[:-1]  # 마지막 행 삭제
        # 마지막 num 값 가져오기
        if not df.empty:
            last_num = df.iloc[-1]['num.']
            num = last_num + 1
        else:
            num = 1
    else:
        df = pd.DataFrame(columns=columns)
        num = 1

    # 새로운 행 추가
    new_row = {'num.': num,
               'train_loss': np.nan,
               'train_mae': np.nan,
               'val_loss': np.nan,
               'val_mae': np.nan,
               'best_val_mae': np.nan,
               'batch_size': args.batch_size,
               'learning_rate': args.learning_rate,
               'epoch': args.epochs,
               'end_epoch': np.nan,
               'target_mae': args.target_mae,
               'target_mae_deviation': args.target_mae_deviation,
               'patience': args.patience,
               'params': np.nan}

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Configuration 정의
    config = {
        'num': num,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'target_mae': args.target_mae,
        'target_mae_deviation': args.target_mae_deviation,
        'patience': args.patience,
    }

    # Initialize WandB
    with open(args.wandb_api, 'r') as f:
        wandb_api_key = f.read().strip()

    wandb.login(key=wandb_api_key)

    # 모델 학습 및 결과 반환
    final_train_loss, final_train_mae, final_val_loss, final_val_mae, best_val_mae, params = train_model(args.target_abbreviation, config)

    # 데이터프레임 업데이트
    df.loc[df['num.'] == num, ['train_loss', 'train_mae', 'val_loss', 'val_mae', 'best_val_mae', 'params', 'end_epoch']] = [
        final_train_loss, final_train_mae, final_val_loss, final_val_mae, best_val_mae, params, args.epochs
    ]

    # 데이터프레임 저장
    df.to_csv(history_file, index=False)
