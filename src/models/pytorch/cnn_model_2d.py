import torch
import torch.nn as nn


class CNNModel_2D(nn.Module):
    def __init__(self, input_shape=(1, 136, 136)):
        super(CNNModel_2D, self).__init__()
        self.input_shape = input_shape

        # 2D Convolutional layers with hardcoded architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=1),  # Conv 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.08712642791973504),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),  # Conv 2
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=3),
            nn.Dropout(0.2637521839803184),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # Conv 3
            nn.ReLU(),
            nn.Dropout(0.14799822825817954),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),  # Conv 4
            nn.ReLU(),
            nn.Dropout(0.04423604806894767),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),  # Conv 5
            nn.ReLU(),
            nn.Dropout(0.2790605562309633),

            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=2),  # Conv 6
            nn.ReLU(),
            nn.Dropout(0.03979758734003716),
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
