import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import wandb
import time
import pynvml  # GPU usage 로깅을 위한 라이브러리

# GPU 모니터링을 위한 NVML 초기화
pynvml.nvmlInit()

# 환경 변수에서 API 키 가져오기
api_key = 'bb88a081ad598db2b81c11dc2621ca6011649c55'  # 보안을 위해 환경 변수를 사용하는 것이 좋습니다

if api_key is None:
    raise ValueError("API 키를 입력하세요")

# API 키를 사용하여 wandb 로그인
wandb.login(key=api_key)

# 프로젝트 및 엔티티로 wandb 초기화
wandb.init(
    project='SePROFiT-Net',         # 프로젝트 이름으로 변경하세요
    name='exp_band_gap',            # 실행(run) 이름으로 변경하세요
    entity='cnmd-phb-postech',      # 팀 이름으로 변경하세요
    config={
        'batch_size': 512,
        'learning_rate': 0.001,
        'epochs': 500,
        'target_mae': 0.5,               # 조기 종료를 위한 목표 MAE
        'target_mae_deviation': 0.05,    # 조기 종료를 위한 목표 MAE 편차
        'patience': 5,                    # 편차를 고려할 Epoch 수
    }
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

batch_size = config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 모델 정의
class MyModel(nn.Module):
    def __init__(self, sequence_length):
        super(MyModel, self).__init__()
        self.sequence_length = sequence_length
        # 합성곱 층
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=21, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=6),
            nn.Dropout(0.01),
            nn.Conv1d(in_channels=21, out_channels=11, kernel_size=9),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=11, out_channels=9, kernel_size=14),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Conv1d(in_channels=9, out_channels=9, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv1d(in_channels=9, out_channels=9, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.01),
        )
        # 합성곱 층 이후의 출력 크기 계산
        self._to_linear = None
        self._get_conv_output()
        # 완전 연결 층
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # 가중치 초기화
        self._initialize_weights()

    def _get_conv_output(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, self.sequence_length)
            output = self.conv_layers(sample_input)
            self._to_linear = output.view(1, -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 평탄화
        x = self.fc_layers(x)
        return x

# 모델 인스턴스 생성
sequence_length = X_train.shape[2]
model = MyModel(sequence_length)

# 가능하면 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# loss 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# 조기 종료를 위한 변수들
target_mae = config.target_mae
target_mae_deviation = config.target_mae_deviation
patience = config.patience
val_mae_history = []
training_stopped = False

# val MAE의 최적 값 추적
best_val_mae = float('inf')
checkpoint_path = 'callback/cp.pt'
os.makedirs('callback', exist_ok=True)

# 학습 루프 (val 및 wandb 로깅 포함)
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    epoch_start_time = time.time()
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

    # val 단계
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

    # val MAE 기록 저장
    val_mae_history.append(val_mae)

    # 최적의 val MAE 확인
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), checkpoint_path)
        print(f'val MAE가 감소하여 모델을 {checkpoint_path}에 저장합니다')
        # 모델 체크포인트를 wandb에 저장
        wandb.save(checkpoint_path)

    # 학습 중지 여부 확인
    if len(val_mae_history) >= patience:
        recent_mae = val_mae_history[-patience:]
        mae_deviation = max(recent_mae) - min(recent_mae)
        if val_mae <= target_mae and mae_deviation <= target_mae_deviation:
            print(f'val MAE가 최근 {patience} Epoch 동안 {target_mae_deviation} 이내로 안정화되었습니다.')
            print('학습을 중지합니다.')
            training_stopped = True

    # 학습 중지 시 루프 탈출
    if training_stopped:
        break

    # GPU usage 로깅
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 단일 GPU를 사용하는 경우
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    # Epoch Duration 시간 계산
    epoch_duration = time.time() - epoch_start_time

    # 현재 learning rate 가져오기
    current_lr = optimizer.param_groups[0]['lr']

    # wandb에 메트릭 로깅
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_mae': train_mae,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'learning_rate': current_lr,
        'gpu_memory_used': mem_info.used / (1024 ** 2),  # MB로 변환
        'gpu_utilization': gpu_util.gpu,  # 퍼센트
        'epoch_duration': epoch_duration,  # sec 단위
    })

    print(f'Epoch {epoch+1}/{config.epochs}, '
          f'train loss: {train_loss:.4f}, train MAE: {train_mae:.4f}, '
          f'val loss: {val_loss:.4f}, val MAE: {val_mae:.4f}, '
          f'learning rate: {current_lr:.6f}, '
          f'GPU memory usage: {mem_info.used / (1024 ** 2):.2f} MB, '
          f'GPU Utilization: {gpu_util.gpu}%, '
          f'Epoch Duration: {epoch_duration:.2f} sec')

# 총 학습 시간 계산
total_time = time.time() - start_time
print(f"총 학습 시간: {total_time:.2f} sec")

# 총 학습 시간을 wandb에 로깅
wandb.log({'total_training_time': total_time})

# wandb 실행 종료
wandb.finish()

# NVML 종료
pynvml.nvmlShutdown()
