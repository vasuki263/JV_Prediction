# model_lstm.py

import torch.nn as nn
import torch

class JV_LSTM(nn.Module):
    # def __init__(self, input_size=10, hidden=64, layers=2):
    #     super().__init__()
    #     self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        # self.fc = nn.Linear(hidden, 3)  # Voc, Jsc, Pmpp
    def __init__(self, input_size=10, hidden=64, layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=3,  # Only V, I, Sweep
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True
        )
        self.fc = nn.Sequential(

            nn.Linear(hidden + 2, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 3)
        )

    # def forward(self, x):
    #     out,_ = self.lstm(x)
    #     return self.fc(out[:,-1,:])
    def forward(self, x):
        seq = x[:, :, :3]

        weather = x[:, 0, 3:5]

        out, _ = self.lstm(seq)

        h = out[:, -1, :]

        h = torch.cat([h, weather], dim=1)

        return self.fc(h)
