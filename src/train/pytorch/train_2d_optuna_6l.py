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

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from src.models.pytorch.cnn_model_2d_optuna import CNNModel_2D
from src.utils.gpu_monitor import log_gpu_usage
from src.utils.target_labels import TargetLabels

def main():
    # 커맨드 라인 인자 파싱
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

    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # WandB API 키 로드 및 로그인
    with open(args.wandb_api, 'r') as f:
        wandb_api_key = f.read().strip()
    wandb.login(key=wandb_api_key)

    # 데이터 로드
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

    # 데이터 텐서로 변환
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()

    # DataLoader 생성
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)

    # 결과 저장을 위한 CSV 파일 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_dir = os.path.join(script_dir, '../../../train_history')
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f'optuna_2DCNN_{args.target_abbreviation}_6layers_independent_channels.csv')

    # Optuna 스터디 생성
    study = optuna.create_study(direction='minimize', study_name=f"optuna_study_{args.target_abbreviation}_6layers_independent_channels")

    # *** 초기 하이퍼파라미터 설정 ***
    initial_heuristic = {
        # 각 합성곱 층의 출력 채널 수
        'c1': 32,
        'c2': 64,
        'c3': 128,
        'c4': 32,
        'c5': 8,
        'c6': 8,

        # 합성곱 층별 하이퍼파라미터
        'kernel_size_0': 5,
        'stride_0': 1,
        'dropout_0': 0.02,
        'padding_0': 0,
        'pool_type_0': 'max',
        'pool_kernel_size_0': 2,
        'pool_stride_0': 2,

        'kernel_size_1': 5,
        'stride_1': 1,
        'dropout_1': 0.03,
        'padding_1': 0,
        'pool_type_1': 'avg',
        'pool_kernel_size_1': 2,
        'pool_stride_1': 2,

        'kernel_size_2': 3,
        'stride_2': 1,
        'dropout_2': 0.02,
        'padding_2': 0,

        'kernel_size_3': 5,
        'stride_3': 2,
        'dropout_3': 0.01,
        'padding_3': 1,

        'kernel_size_4': 3,
        'stride_4': 1,
        'dropout_4': 0.01,
        'padding_4': 0,

        'kernel_size_5': 3,
        'stride_5': 1,
        'dropout_5': 0.01,
        'padding_5': 0,
    }

    # 초기 하이퍼파라미터를 큐에 추가
    study.enqueue_trial(initial_heuristic)

    # Objective 함수 정의
    def objective(trial):
        repeat_mae_list = []

        for repeat_idx in range(args.repeats):
            # WandB 실행 초기화
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

            # 하이퍼파라미터 설정
            learning_rate = args.learning_rate
            epochs = args.epochs

            # 합성곱 층 수 설정 (6개)
            num_conv_layers = 6

            conv_params_list = []

            # 각 합성곱 층의 출력 채널 수를 독립적으로 정의
            c1 = trial.suggest_categorical('c1', [16, 32, 64])
            c2 = trial.suggest_categorical('c2', [16, 32, 64, 128, 256])
            c3 = trial.suggest_categorical('c3', [32, 64, 128, 256, 512])
            c4 = trial.suggest_categorical('c4', [32, 64, 128, 256])
            c5 = trial.suggest_categorical('c5', [8, 16, 32, 64])
            c6 = trial.suggest_categorical('c6', [8, 16])

            output_channels_list = [c1, c2, c3, c4, c5, c6]

            for i in range(num_conv_layers):
                conv_params = {}

                conv_params['out_channels'] = output_channels_list[i]

                # 커널 크기
                conv_params['kernel_size'] = trial.suggest_categorical(f'kernel_size_{i}', [3, 5])

                # 스트라이드
                conv_params['stride'] = trial.suggest_int(f'stride_{i}', 1, 2)

                # 드롭아웃 확률
                conv_params['dropout'] = trial.suggest_float(f'dropout_{i}', 0.0, 0.3)

                # 패딩
                conv_params['padding'] = trial.suggest_int(f'padding_{i}', 0, 2)

                # 첫 두 개의 층에 풀링 층 추가
                if i < 2:
                    conv_params['pooling'] = True
                    conv_params['pool_type'] = trial.suggest_categorical(f'pool_type_{i}', ['max', 'avg'])
                    conv_params['pool_kernel_size'] = trial.suggest_int(f'pool_kernel_size_{i}', 2, 3)
                    conv_params['pool_stride'] = trial.suggest_int(f'pool_stride_{i}', 2, 3)
                else:
                    conv_params['pooling'] = False

                conv_params_list.append(conv_params)

            # 모델 생성
            try:
                model = CNNModel_2D(conv_params_list).to(device)
            except Exception as e:
                print(f"Error in model creation: {e}")
                wandb.finish()
                return float('inf')

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            params = sum(p.numel() for p in model.parameters())  # 파라미터 수 계산

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

                # 검증
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

                # WandB에 로그 기록
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

                # 최상의 검증 MAE 업데이트
                if val_mae < best_val_mae:
                    best_val_mae = val_mae

            # 반복의 MAE 저장
            repeat_mae_list.append(best_val_mae)

            # WandB 실행 종료
            wandb.finish()

        # 반복의 평균 MAE 계산
        mean_val_mae = np.mean(repeat_mae_list)

        # 결과 저장
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

        return mean_val_mae  # 최적화 목표값

    # 최적화 실행
    study.optimize(objective, n_trials=args.trials)

    # 스터디 결과 저장
    study.trials_dataframe().to_csv(os.path.join(history_dir, f'study_2DCNN_{args.target_abbreviation}_6layers_independent_channels.csv'), index=False)

if __name__ == "__main__":
    main()
