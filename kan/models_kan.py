"""KAN (Kolmogorov-Arnold Network) para classificação binária.

Implementação completa no estilo da "efficient-kan" (Blealtan, licença MIT,
https://github.com/Blealtan/efficient-kan), seguindo a formulação de
Liu et al. (2024), "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756.

Cada conexão entrada -> saída tem uma função aprendível

    phi(x) = w_base * act(x) + s * sum_m c_m * B_m(x)

- w_base * act(x): parte base (ativação fixa, ex. SiLU, com peso aprendível)
- c_m * B_m(x):    parte spline (B-splines num grid, coeficientes aprendíveis)
- s:               spline_scaler, escala aprendível da parte spline por conexão


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
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps  # 0 = grid só por quantis dos dados, 1 = grid uniforme

        # Grid uniforme inicial, estendido por `spline_order` nós de cada lado
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())  # (in, G + 2k + 1)

        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Splines começam aproximando um ruído pequeno nos pontos do grid
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            coeff = self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            scale = 1.0 if self.enable_standalone_scale_spline else self.scale_spline
            self.spline_weight.data.copy_(scale * coeff)
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        """Bases B-spline de cada entrada. x: (B, in) -> (B, in, G + k)."""
        grid = self.grid
        x = x.unsqueeze(-1)
        # ordem 0: indicadora do intervalo [g_j, g_{j+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # recursão de Cox-de Boor
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            bases = left + right
        return bases.contiguous()

    def curve2coeff(self, x, y):
        """Coeficientes c tais que sum_m c_m B_m(x) ~= y (mínimos quadrados).

        x: (B, in) | y: (B, in, out) -> (out, in, G + k)
        """
        A = self.b_splines(x).transpose(0, 1)  # (in, B, coeff)
        B = y.transpose(0, 1)                  # (in, B, out)
        # CPU + float64 + gelsd: estável e aceita sistemas sem posto completo
        solution = torch.linalg.lstsq(A.double().cpu(), B.double().cpu(), driver="gelsd").solution
        result = solution.permute(2, 0, 1).to(dtype=x.dtype, device=x.device)  # (out, in, coeff)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x):
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return base_out + spline_out

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        """Reposiciona os nós do grid conforme a distribuição de `x` (B, in) e
        reajusta os coeficientes para a função aprendida continuar a mesma."""
        batch = x.size(0)

        # 1. Saída atual da parte spline, por conexão (alvo do reajuste)
        splines = self.b_splines(x).permute(1, 0, 2)              # (in, B, coeff)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)   # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)  # (B, in, out)

        # 2. Novo grid: mistura de quantis dos dados (adaptativo) e grid uniforme
        x_sorted = torch.sort(x, dim=0)[0]
        idx = torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        grid_adaptive = x_sorted[idx]                                              # (G+1, in)
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size  # (in,)
        steps = torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
        grid_uniform = steps * uniform_step + x_sorted[0] - margin                 # (G+1, in)
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # Colunas com poucos valores distintos (ex.: dummies 0/1) -> grid uniforme
        n_unique = (x_sorted[1:] != x_sorted[:-1]).sum(dim=0) + 1                 # (in,)
        few_values = n_unique <= self.grid_size
        grid = torch.where(few_values.unsqueeze(0), grid_uniform, grid)

        # Extensão de `spline_order` nós de cada lado
        k = self.spline_order
        grid = torch.cat(
            [
                grid[:1] - uniform_step * torch.arange(k, 0, -1, dtype=x.dtype, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, k + 1, dtype=x.dtype, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)

        # 3. Reajusta os coeficientes no grid novo para reproduzir a mesma saída
        new_coeff = self.curve2coeff(x, unreduced_spline_output)  # já inclui a escala
        if self.enable_standalone_scale_spline:
            scaler = self.spline_scaler.unsqueeze(-1)
            safe = torch.where(scaler.abs() < 1e-8, torch.full_like(scaler, 1e-8), scaler)
            new_coeff = new_coeff / safe
        self.spline_weight.data.copy_(new_coeff)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Regularização da KAN: L1 das "ativações" (aproximada pelos coeficientes)
        + entropia da distribuição entre conexões. Incentiva poucas conexões ativas."""
        l1_fake = self.spline_weight.abs().mean(-1)          # (out, in)
        reg_activation = l1_fake.sum()
        p = l1_fake / (reg_activation + 1e-12)
        reg_entropy = -torch.sum(p * torch.log(p + 1e-12))
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy


class KAN(nn.Module):
    def __init__(self, layers, grid_size=5, spline_order=3, base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1.0, 1.0)):
        """layers: tamanhos das camadas, ex.: [45, 16, 1]."""
        super().__init__()
        self.layers = nn.ModuleList(
            KANLinear(
                n_in, n_out,
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            for n_in, n_out in zip(layers[:-1], layers[1:])
        )

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)  # cada camada ajusta o grid à entrada que ela recebe
            x = layer(x)
        return x.squeeze(-1)  # (B,) -> logits

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)


def build_kan(input_dim, num_hidden_layers=1, hidden_dim=16, grid_size=5, spline_order=3, activation_name="SiLU"):
    layers = [input_dim] + [hidden_dim] * num_hidden_layers + [1]
    return KAN(layers, grid_size=grid_size, spline_order=spline_order, base_activation=ACTIVATIONS[activation_name])


@torch.no_grad()
def update_model_grid(model, loader, device, max_samples=2048):
    """Atualiza o grid de todas as camadas com uma amostra do conjunto de treino."""
    x = loader.dataset.tensors[0]
    if x.size(0) > max_samples:
        idx = torch.randperm(x.size(0))[:max_samples]
        x = x[idx]
    model.eval()
    model(x.to(device), update_grid=True)


def train_epoch(model, loader, optimizer, criterion, device, reg_lambda=0.0, update_grid=False):
    """Uma época de treino.

    reg_lambda:  peso da regularização da KAN somada à loss (0 = desligada)
    update_grid: se True, reposiciona o grid conforme os dados antes da época
    """
    if update_grid:
        update_model_grid(model, loader, device)

    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        bce = criterion(model(x), y)
        loss = bce + reg_lambda * model.regularization_loss() if reg_lambda > 0 else bce
        loss.backward()
        optimizer.step()
        total_loss += bce.item() * x.size(0)  # registra só a BCE, comparável entre trials
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