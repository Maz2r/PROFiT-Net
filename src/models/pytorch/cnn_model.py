# src/models/pytorch/cnn_model.py

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, sequence_length):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 21, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool1d(6),
            nn.Dropout(0.01),
            nn.Conv1d(21, 11, kernel_size=9),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.MaxPool1d(2),
            nn.Conv1d(11, 9, kernel_size=14),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Conv1d(9, 9, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv1d(9, 9, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.01),
        )
        self._to_linear = None
        self._get_conv_output(sequence_length)
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

    def _get_conv_output(self, sequence_length):
        with torch.no_grad():
            x = torch.zeros(1, 1, sequence_length)
            output = self.conv_layers(x)
            self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
