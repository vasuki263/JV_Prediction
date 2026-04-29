# model_lstm.py

import torch
import torch.nn as nn

class VocJscLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)
