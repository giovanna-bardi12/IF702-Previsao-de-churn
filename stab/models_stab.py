# Um token por coluna e modulação de canais
import time

import torch
from torch import nn


class STabBlock(nn.Module):  
    #monta um bloco de atenção multi-cabeça com normalização e projeção
    def __init__(self, n_features, d_model, n_heads, ffn_multiplier, gate_hidden, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) 
        self.attention_norm = nn.LayerNorm(d_model)  
        self.projection = nn.Sequential( 
            nn.Linear(d_model, ffn_multiplier * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * d_model, d_model),
        )
        self.gate = nn.Sequential( 
            nn.Linear(n_features * d_model, gate_hidden), nn.GELU(),
            nn.Linear(gate_hidden, n_features * d_model), nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.n_features = n_features
        self.d_model = d_model

    # Recebe um tensor de tokens (batch, n_features, d_model) e devolve um tensor do mesmo shape.
    def forward(self, tokens):
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + self.dropout(attended)) #preserva a entrada e soma o que o token aprendeu com a atenção
        gates = self.gate(tokens.flatten(1)).reshape(-1, self.n_features, self.d_model)
        return self.output_norm(tokens + gates * self.projection(tokens))


class STabModel(nn.Module):
 # Recebe o vetor numérico já preparado e devolve um logit de churn.

    def __init__(self, n_features, d_model=16, n_heads=4, n_layers=1,
                 ffn_multiplier=4, gate_hidden=64, classifier_hidden=32, dropout=0.1):
        super().__init__()
        if min(n_features, n_layers, d_model, n_heads) < 1 or d_model % n_heads:
            raise ValueError("Dimensões devem ser positivas e d_model divisível por n_heads")
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.feature_embedding = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.feature_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            STabBlock(n_features, d_model, n_heads, ffn_multiplier, gate_hidden, dropout)
            for _ in range(n_layers)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1), nn.Linear(n_features * d_model, classifier_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias + self.feature_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self.classifier(tokens).squeeze(-1)


def build_stab(**kwargs):
    return STabModel(**kwargs)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_time = 0.0, 0.0
    labels, probabilities = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_loss += criterion(logits, y).item() * len(y)
        labels.append(y.cpu())
        probabilities.append(logits.sigmoid().cpu())
    return (total_loss / len(loader.dataset), total_time / len(loader),
            torch.cat(labels).numpy().astype(int), torch.cat(probabilities).numpy())
