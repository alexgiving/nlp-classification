from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomModel(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 vocab_size: int,
                 num_classes: int,
                 num_rnn_layers: int,
                 p_drop_out: float=0.1) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim

        self._embedding = nn.Embedding(vocab_size, hidden_dim)
        self._linear = nn.Linear(hidden_dim, hidden_dim)
        self._classifier = nn.Linear(hidden_dim, num_classes)
        self._activation = nn.Tanh()
        self._dropout = nn.Dropout(p=p_drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._embedding(x)
        output, _ = self._rnn(output)
        output = output.mean(dim=1)
        output = self._activation(output)
        output = self._linear(output)
        output = self._dropout(output)
        output = self._activation(output)
        return self._classifier(output)


class RNNModel(CustomModel):
    def __init__(self,
                 hidden_dim: int,
                 vocab_size: int,
                 num_classes: int,
                 num_rnn_layers: int,
                 p_drop_out: float=0.1) -> None:
        super().__init__(
            hidden_dim,
            vocab_size,
            num_classes,
            num_rnn_layers,
            p_drop_out
        )
        self._rnn = nn.RNN(
            self._hidden_dim,
            self._hidden_dim,
            num_layers=num_rnn_layers
        )


class LSTMModel(CustomModel):
    def __init__(self,
                 hidden_dim: int,
                 vocab_size: int,
                 num_classes: int,
                 num_rnn_layers: int,
                 p_drop_out: float=0.1) -> None:
        super().__init__(
            hidden_dim,
            vocab_size,
            num_classes,
            num_rnn_layers,
            p_drop_out
        )
        self._rnn = nn.LSTM(
            self._hidden_dim,
            self._hidden_dim,
            num_layers=num_rnn_layers
        )

def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
    losses = []

    for train_batch in tqdm(data_loader):
        optimizer.zero_grad()

        res = model(train_batch['input_ids'].to(device))
        loss = criterion(res, train_batch['label'])
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().item())
    return np.mean(losses)

def evaluate(model, eval_dataloader) -> float:
    predictions = []
    target = []
    with torch.no_grad():
        for batch in eval_dataloader:
            logits = model(batch['input_ids'])
            predictions.append(logits.argmax(dim=1))
            target.append(batch['label'])

    predictions = torch.cat(predictions)
    target = torch.cat(target)
    accuracy = (predictions == target).float().mean().item()

    return accuracy

def train(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[List[float], List[float]]:
    train_losses = []
    accuracies = []

    for epoch_id in range(num_epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        model.eval()
        with torch.inference_mode():
            accuracy = evaluate(model, val_loader)
        accuracies.append(accuracy)
    return train_losses, accuracies

