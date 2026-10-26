"""KAN (Kolmogorov-Arnold Network) para classificação binária.

Implementação própria, no estilo da "efficient-kan" (Blealtan, licença MIT):
cada conexão entrada->saída tem uma função aprendível

    phi(x) = w_base * act(x) + sum_i c_i * B_i(x)
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"SiLU": nn.SiLU, "ReLU": nn.ReLU, "GELU": nn.GELU}


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
        grid_range=(-1.0, 1.0),
        scale_noise=0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid uniforme estendido por `spline_order` nós de cada lado
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())  # (in, G + 2k + 1)

        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise / grid_size)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        # ordem 0: indicadora do intervalo [g_j, g_{j+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # recursão de Cox-de Boor
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k]) * bases[:, :, 1:]
            bases = left + right
        return bases

    def forward(self, x):
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.spline_weight.reshape(self.out_features, -1),
        )
        return base_out + spline_out


class KAN(nn.Module):
    def __init__(self, layers, grid_size=5, spline_order=3, base_activation=nn.SiLU):
        """layers: tamanhos das camadas, ex.: [45, 16, 1]."""
        super().__init__()
        self.layers = nn.ModuleList(
            KANLinear(n_in, n_out, grid_size=grid_size, spline_order=spline_order, base_activation=base_activation)
            for n_in, n_out in zip(layers[:-1], layers[1:])
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)  # (B,) -> logits


def build_kan(input_dim, num_hidden_layers=1, hidden_dim=16, grid_size=5, spline_order=3, activation_name="SiLU"):
    layers = [input_dim] + [hidden_dim] * num_hidden_layers + [1]
    return KAN(layers, grid_size=grid_size, spline_order=spline_order, base_activation=ACTIVATIONS[activation_name])


def train_epoch(model, loader, optimizer, criterion, device):
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