import torch
import torch.nn as nn

class CNNModel_2D(nn.Module):
    def __init__(self, conv_params_list, input_shape=(1, 136, 136)):
        super(CNNModel_2D, self).__init__()
        self.input_shape = input_shape

        layers = []
        in_channels = 1  # 입력 채널 수는 1로 고정

        for idx, conv_params in enumerate(conv_params_list):
            # 합성곱 층 추가
            out_channels = conv_params['out_channels']
            kernel_size = conv_params['kernel_size']
            stride = conv_params['stride']
            padding = conv_params.get('padding', 0)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())

            # 풀링 층 추가
            if conv_params.get('pooling', False):
                pool_type = conv_params['pool_type']
                pool_kernel_size = conv_params['pool_kernel_size']
                pool_stride = conv_params['pool_stride']
                if pool_type == 'max':
                    layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
                elif pool_type == 'avg':
                    layers.append(nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride))

            # 드롭아웃 추가
            dropout_prob = conv_params.get('dropout', 0.0)
            if dropout_prob > 0.0:
                layers.append(nn.Dropout2d(dropout_prob))

            in_channels = out_channels  # 다음 층의 입력 채널 수 업데이트

        self.conv_layers = nn.Sequential(*layers)

        # 합성곱 층 이후의 출력 크기 계산
        self._to_linear = self._get_conv_output()

        # Fully connected layers (기존과 동일)
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

        self._initialize_weights()

    def _get_conv_output(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, *self.input_shape)
            output = self.conv_layers(sample_input)
            output_size = output.view(1, -1).size(1)
            if output_size == 0:
                raise ValueError("The output size after convolutional layers is zero. Adjust the parameters.")
            return output_size

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
