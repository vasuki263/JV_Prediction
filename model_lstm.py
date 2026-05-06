# model_lstm.py

import torch.nn as nn
import torch

class JV_LSTM(nn.Module):
    def __init__(self, input_size=17, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 3)  # Voc, Jsc, Pmpp

    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])
