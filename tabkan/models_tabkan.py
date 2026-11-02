import time
import torch
import torch.nn as nn
from tabkan import KANMixer, ChebyshevKAN


class ChebyshevKANLayerWrapper(nn.Module):
    """Adapta a chamada posicional do KANMixer:
        kan_layer_class(in_features, out_features, **kwargs)
    para o formato esperado pelo ChebyshevKAN:
        ChebyshevKAN(layers=[in_features, out_features], orders=poly_order)
    """
    def __init__(self, in_features: int, out_features: int, order: int = 3, **kwargs):
        super().__init__()
        # Captura a ordem dos polinômios seja via 'order' ou 'orders'
        poly_order = kwargs.get("order", kwargs.get("orders", order))
        self.kan = ChebyshevKAN(layers=[in_features, out_features], orders=poly_order)

    def forward(self, x):
        return self.kan(x)


class TabKANWrapper(nn.Module):
    """Wrapper para adaptar o KANMixer ao pipeline PyTorch e BCEWithLogitsLoss."""
    def __init__(
        self, 
        num_features: int, 
        num_layers: int = 2, 
        token_dim: int = 16, 
        channel_dim: int = 32, 
        token_order: int = 3, 
        channel_order: int = 3
    ):
        super().__init__()

        self.model = KANMixer(
            num_features=num_features,
            num_classes=1,                             # 1 logit de saída (classificação binária)
            kan_layer_class=ChebyshevKANLayerWrapper, # Usa o wrapper adaptador
            num_layers=num_layers,
            token_dim=token_dim,
            channel_dim=channel_dim,
            token_kan_kwargs={"order": token_order},
            channel_kan_kwargs={"order": channel_order},
        )

    def forward(self, x):
        logits = self.model(x)
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits


def train_epoch(model, loader, optimizer, criterion, device):
    """Executa uma época de treinamento do modelo."""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """Valida o modelo e calcula perda, tempo de inferência, rótulos e probabilidades."""
    model.eval()
    total_loss = 0.0
    times = []
    all_labels, all_probs = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)

        start = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits)
        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())

    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.cat(all_probs).numpy()
    avg_inference_time = sum(times) / len(times) if times else 0.0
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, avg_inference_time, labels, probs