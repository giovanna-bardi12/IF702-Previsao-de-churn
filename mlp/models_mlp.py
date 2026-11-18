"""MLP (Multilayer Perceptron) para classificação binária.

Arquitetura: [Linear -> (BatchNorm) -> ativação -> (Dropout)] x num_hidden_layers -> Linear(1)

A saída é um logit (sem sigmoid), para usar com BCEWithLogitsLoss — igual à KAN,
assim train_epoch / validate_epoch / métricas têm exatamente a mesma interface.
"""
import time

import torch
import torch.nn as nn

ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU, dropout=0.0, batch_norm=False):
        """layers: tamanhos das camadas, ex.: [45, 16, 1]."""
        super().__init__()
        blocks = []
        for n_in, n_out in zip(layers[:-2], layers[1:-1]):  # camadas ocultas
            blocks.append(nn.Linear(n_in, n_out))
            if batch_norm:
                blocks.append(nn.BatchNorm1d(n_out))
            blocks.append(activation())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(layers[-2], layers[-1]))   # camada de saída (logit)
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) -> logits


def build_mlp(input_dim, num_hidden_layers=1, hidden_dim=16, activation_name="ReLU", dropout=0.0, batch_norm=False):
    layers = [input_dim] + [hidden_dim] * num_hidden_layers + [1]
    return MLP(layers, activation=ACTIVATIONS[activation_name], dropout=dropout, batch_norm=batch_norm)


def train_epoch(model, loader, optimizer, criterion, device):
    """Uma época de treino. Retorna a BCE média no conjunto de treino."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """Retorna (loss médio, tempo médio de inferência por batch, rótulos, probabilidades)."""
    model.eval()
    total_loss, times = 0.0, []
    all_labels, all_probs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        start = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        total_loss += criterion(logits, y).item() * x.size(0)
        all_labels.append(y.cpu())
        all_probs.append(torch.sigmoid(logits).cpu())

    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.cat(all_probs).numpy()
    return total_loss / len(loader.dataset), sum(times) / len(times), labels, probs
