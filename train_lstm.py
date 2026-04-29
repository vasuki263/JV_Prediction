# train_lstm.py

import torch
import torch.nn as nn
from model_lstm import JV_LSTM
from build_dataset import build_dataset
import matplotlib.pyplot as plt

def train(directory):
    loader, scalers = build_dataset(directory)
    model = JV_LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []

    for epoch in range(40):
        running = 0
        for X,Y in loader:
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred,Y)
            loss.backward()
            opt.step()
            running += loss.item()
        losses.append(running/len(loader))
        print(f"Epoch {epoch+1} | Loss {losses[-1]:.5f}")

    torch.save({"model":model.state_dict(),"scalers":scalers},"jv_model.pth")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.grid()
    plt.savefig("training_curve.png")
    plt.close()

if __name__ == "__main__":
    train("C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw")

