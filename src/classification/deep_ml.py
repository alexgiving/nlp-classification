from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from torchmetrics.functional.classification import accuracy, f1_score
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


def prepare_dataset(x: pd.Series, y: pd.Series, tokenizer: AutoTokenizer) -> Dataset:
    tokenized_inputs = tokenizer(list(x), padding='max_length', truncation=True, return_tensors='pt')
    dataset = Dataset.from_dict(
        {
            'input_ids': tokenized_inputs['input_ids'],
            'label': torch.tensor(y, dtype=torch.long)
        }
    )
    return dataset


def compute_metrics(pred: Tuple[np.typing.NDArray, np.typing.NDArray], num_classes: int = 2) -> Dict[str, float]:
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy(predictions, torch.tensor(labels), 'multiclass', num_classes=num_classes)
    f1 = f1_score(predictions, torch.tensor(labels), 'multiclass', num_classes=num_classes)
    return {'accuracy': acc.item(), 'f1': f1}


class BaseClassifier(ABC, nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.backbone = self._get_backbone(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids=None, labels=None, **kwargs):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.backbone(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        logits = self.fc(self.dropout(last_hidden_state))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @abstractmethod
    def _get_backbone(self, *args, **kwargs) -> nn.Module:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class LSTMClassifier(BaseClassifier):
    def _get_backbone(self, *args, **kwargs) -> nn.Module:
        return nn.LSTM(kwargs['embedding_dim'], kwargs['hidden_dim'], num_layers=kwargs['num_layers'], batch_first=True)


class RNNClassifier(BaseClassifier):
    def _get_backbone(self, *args, **kwargs) -> nn.Module:
        return nn.RNN(kwargs['embedding_dim'], kwargs['hidden_dim'], num_layers=kwargs['num_layers'], batch_first=True)
