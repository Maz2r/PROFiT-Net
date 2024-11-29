import torch
import torch.nn as nn


class CNNModel_2D(nn.Module):
    def __init__(self, input_shape=(1, 136, 136)):
        super(CNNModel_2D, self).__init__()
        self.input_shape = input_shape

        # 2D Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.02),
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

        # Calculate the output size after convolutional layers
        self._to_linear = self._get_conv_output()

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

        self._initialize_weights()

    def _get_conv_output(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, *self.input_shape)
            output = self.conv_layers(sample_input)
            return output.view(1, -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
