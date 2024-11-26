import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import wandb
import time
import pynvml  # GPU 사용량 로깅을 위한 라이브러리
import optuna
from optuna.trial import TrialState

target_prop = "exp_band_gap"

# GPU 모니터링을 위한 NVML 초기화
pynvml.nvmlInit()

# 환경 변수에서 API 키 가져오기
api_key = 'bb88a081ad598db2b81c11dc2621ca6011649c55'

if api_key is None:
    raise ValueError("API 키를 입력하세요")

# API 키를 사용하여 wandb 로그인
wandb.login(key=api_key)

# 프로젝트 및 엔티티로 wandb 초기화
wandb.init(
    project='SePROFiT-Net',         # 프로젝트 이름으로 변경하세요
    name='exp_band_gap_optuna',     # 실행(run) 이름으로 변경하세요
    entity='cnmd-phb-postech',      # 팀 이름으로 변경하세요
)
config = wandb.config

# 실행 시간 추적 시작
start_time = time.time()

# 데이터 로드
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# 데이터를 PyTorch 텐서로 변환하고 채널 차원 추가
X_train = torch.from_numpy(X_train).float().unsqueeze(1)  # 형태: (샘플 수, 1, 시퀀스 길이)
y_train = torch.from_numpy(y_train).float()
X_val = torch.from_numpy(X_val).float().unsqueeze(1)
y_val = torch.from_numpy(y_val).float()

# 데이터셋 및 데이터로더 생성
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# 가능하면 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 최적화를 위한 objective 함수 정의
def objective(trial):
    # 고정된 하이퍼파라미터 설정
    learning_rate = 0.001  # 고정된 학습률
    num_epochs = 50        # 에포크 수를 적게 설정하여 빠른 실험

    # 하이퍼파라미터 샘플링

    # 합성곱 층 관련 하이퍼파라미터
    n_conv_layers = 5  # 합성곱 층 수 고정

    conv_channels = []
    kernel_sizes = []
    dropout_rates = []

    for i in range(n_conv_layers):
        # out_channels 최적화 (서서히 감소하도록 설정)
        if i == 0:
            out_channels = trial.suggest_int(f'conv_{i}_out_channels', 8, 32)
        else:
            # 이전 층의 out_channels을 참조하여 현재 층의 out_channels가 작거나 같도록 설정
            out_channels = trial.suggest_int(f'conv_{i}_out_channels', 8, conv_channels[i-1])
        conv_channels.append(out_channels)

        # kernel_size 최적화
        if i == n_conv_layers - 1:
            kernel_size = 1  # 다섯번째 Conv1D의 kernel_size는 1로 고정
        elif i == 2:
            # 세번째 Conv1D의 kernel_size는 8에서 20 사이로 샘플링
            kernel_size = trial.suggest_int(f'conv_{i}_kernel_size', 8, 20)
        elif i == 3:
            # 네번째 Conv1D의 kernel_size는 세번째 Conv1D의 kernel_size보다 작게 샘플링
            min_kernel = 3
            max_kernel = kernel_sizes[2] - 1  # 세번째 Conv1D의 kernel_size보다 작아야 함
            if max_kernel < 3:
                # 제약 조건을 만족할 수 없을 경우 trial을 프루닝
                raise optuna.exceptions.TrialPruned()
            kernel_size = trial.suggest_int(f'conv_{i}_kernel_size', min_kernel, max_kernel)
        else:
            # 첫번째, 두번째, 네번째 Conv1D의 kernel_size는 3에서 14 사이로 샘플링
            kernel_size = trial.suggest_int(f'conv_{i}_kernel_size', 3, 14)
        kernel_sizes.append(kernel_size)

        # 드롭아웃 비율 최적화 (각 층마다, 0.0에서 0.1 이하)
        dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.1)
        dropout_rates.append(dropout_rate)

    # 활성화 함수 선택
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU'])
    activation = getattr(nn, activation_name)()

    # 첫번째 풀링 층 하이퍼파라미터
    pooling_method_1 = trial.suggest_categorical('pooling_method_1', ['MaxPool1d', 'AvgPool1d'])
    pooling_kernel_size_1 = trial.suggest_int('pooling_kernel_size_1', 2, 8)

    # 두번째 풀링 층 하이퍼파라미터
    pooling_method_2 = trial.suggest_categorical('pooling_method_2', ['MaxPool1d', 'AvgPool1d'])
    pooling_kernel_size_2 = trial.suggest_int('pooling_kernel_size_2', 2, 8)

    # 옵티마이저 선택
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    # 배치 크기 선택
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # 모델 정의
    class MyModel(nn.Module):
        def __init__(self, sequence_length):
            super(MyModel, self).__init__()
            self.sequence_length = sequence_length

            layers = []
            in_channels = 1
            seq_len = sequence_length

            for i in range(n_conv_layers):
                out_channels = conv_channels[i]
                kernel_size = kernel_sizes[i]
                dropout_rate = dropout_rates[i]

                # 합성곱 층
                conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
                layers.append(conv_layer)
                layers.append(activation)

                # 시퀀스 길이 업데이트 (Conv1d)
                conv_stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
                conv_padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
                conv_dilation = conv_layer.dilation[0] if isinstance(conv_layer.dilation, tuple) else conv_layer.dilation

                seq_len = (seq_len + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1

                # 풀링 층 적용
                if i == 0:
                    pooling_layer = getattr(nn, pooling_method_1)(kernel_size=pooling_kernel_size_1)
                    layers.append(pooling_layer)

                    # 시퀀스 길이 업데이트 (Pooling)
                    pooling_kernel_size_actual = pooling_layer.kernel_size
                    if isinstance(pooling_kernel_size_actual, tuple):
                        pooling_kernel_size_actual = pooling_kernel_size_actual[0]

                    pooling_stride_actual = pooling_layer.stride
                    if pooling_stride_actual is None:
                        pooling_stride_actual = pooling_kernel_size_actual
                    elif isinstance(pooling_stride_actual, tuple):
                        pooling_stride_actual = pooling_stride_actual[0]

                    pooling_padding_actual = pooling_layer.padding
                    if isinstance(pooling_padding_actual, tuple):
                        pooling_padding_actual = pooling_padding_actual[0]

                    seq_len = (seq_len + 2 * pooling_padding_actual - (pooling_kernel_size_actual - 1) - 1) // pooling_stride_actual + 1

                elif i == 1:
                    pooling_layer = getattr(nn, pooling_method_2)(kernel_size=pooling_kernel_size_2)
                    layers.append(pooling_layer)

                    # 시퀀스 길이 업데이트 (Pooling)
                    pooling_kernel_size_actual = pooling_layer.kernel_size
                    if isinstance(pooling_kernel_size_actual, tuple):
                        pooling_kernel_size_actual = pooling_kernel_size_actual[0]

                    pooling_stride_actual = pooling_layer.stride
                    if pooling_stride_actual is None:
                        pooling_stride_actual = pooling_kernel_size_actual
                    elif isinstance(pooling_stride_actual, tuple):
                        pooling_stride_actual = pooling_stride_actual[0]

                    pooling_padding_actual = pooling_layer.padding
                    if isinstance(pooling_padding_actual, tuple):
                        pooling_padding_actual = pooling_padding_actual[0]

                    seq_len = (seq_len + 2 * pooling_padding_actual - (pooling_kernel_size_actual - 1) - 1) // pooling_stride_actual + 1

                # 드롭아웃 층
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                in_channels = out_channels

                # 시퀀스 길이가 음수 또는 0이 되는 경우 처리
                if seq_len <= 0:
                    raise optuna.exceptions.TrialPruned()

            self.conv_layers = nn.Sequential(*layers)

            # 합성곱 층 이후의 출력 크기 계산
            self._to_linear = in_channels * seq_len

            # 완전 연결 층
            self.fc_layers = nn.Sequential(
                nn.Linear(self._to_linear, 1024),
                activation,
                nn.Linear(1024, 256),
                activation,
                nn.Linear(256, 512),
                activation,
                nn.Linear(512, 64),
                activation,
                nn.Linear(64, 16),
                activation,
                nn.Linear(16, 1)
            )

            # 가중치 초기화
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # 평탄화
            x = self.fc_layers(x)
            return x

    sequence_length = X_train.shape[2]

    try:
        model = MyModel(sequence_length).to(device)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"모델 생성 중 에러 발생: {e}")
        raise optuna.exceptions.TrialPruned()

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # 학습률 스케줄러 (옵션)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 학습 루프
    best_val_mae = float('inf')
    for epoch in range(num_epochs):
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

        # 검증 단계
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

        # 학습률 스케줄러 업데이트
        scheduler.step(val_mae)

        # 최적의 검증 MAE 업데이트
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        # 조기 종료 조건 (시간 절약을 위해)
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_mae

# Optuna 스터디 생성 및 실행
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)

print('최적의 하이퍼파라미터:')
for key, value in study.best_params.items():
    print(f'{key}: {value}')

print(f'최소 검증 MAE: {study.best_value}')

# 최적의 하이퍼파라미터로 모델 재학습은 별도의 파이썬 파일에서 진행

# wandb 실행 종료
wandb.finish()

# NVML 종료
pynvml.nvmlShutdown()
