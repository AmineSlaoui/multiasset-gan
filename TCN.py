import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, bias: bool = True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation          # left-only causal pad
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, channels, time)
        x = nn.functional.pad(x, (self.padding, 0))          # causal left-pad
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.causal1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.causal2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.conv1 = weight_norm(self.causal1.conv)
        self.conv2 = weight_norm(self.causal2.conv)

        self.relu1    = nn.ReLU()
        self.relu2    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv1, self.conv2]:
            nn.init.normal_(conv.weight_v, mean=0.0, std=0.5)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)

        # causal1 / causal2 handle the left-padding; conv1/conv2 are the weight-normed inner convs
        out = nn.functional.pad(x,   (self.causal1.padding, 0))
        out = self.dropout1(self.relu1(self.conv1(out)))
        out = nn.functional.pad(out, (self.causal2.padding, 0))
        out = self.dropout2(self.relu2(self.conv2(out)))

        return out + residual                                 # eq. (6) in paper

class TCN(nn.Module):
    def __init__(
        self,
        in_channels:      int,
        out_channels:     int,
        hidden_channels:  int,
        num_blocks:       int   = 6,
        kernel_size:      int   = 2,
        dilation_base:    int   = 2,
        dropout:          float = 0.2,
    ):
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            dilation = dilation_base ** i
            blocks.append(
                TemporalBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    dropout      = dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    @staticmethod
    def receptive_field_size(kernel_size: int, dilation_base: int, num_blocks: int) -> int:
        return 1 + 2 * (kernel_size - 1) * (dilation_base ** num_blocks - 1) // (dilation_base - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x)   # (B, hidden, T)
        out = self.blocks(out)     # (B, hidden, T)
        out = self.output_proj(out)# (B, out_channels, T)
        return out


# ---------------------------------------------------------------------------
# Generator TCN  (80 hidden channels)
# ---------------------------------------------------------------------------

class GeneratorTCN(nn.Module):
    def __init__(
        self,
        in_channels:     int,
        out_alpha:       int,
        out_beta:        int,
        out_sigma:       int,
        hidden_channels: int   = 80,
        num_blocks:      int   = 6,
        kernel_size:     int   = 2,
        dilation_base:   int   = 2,
        dropout:         float = 0.2,
    ):
        super().__init__()

        # ϕ_I  (shared)
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # g_1 … g_L  (shared)
        blocks = []
        for i in range(num_blocks):
            dilation = dilation_base ** i
            blocks.append(
                TemporalBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    dropout      = dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Three separate ϕ_O heads  (NOT shared)
        self.alpha_proj = nn.Conv1d(hidden_channels, out_alpha, kernel_size=1)
        self.beta_proj  = nn.Conv1d(hidden_channels, out_beta,  kernel_size=1)
        self.sigma_proj = nn.Conv1d(hidden_channels, out_sigma, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, 0.0, 0.5)
        nn.init.zeros_(self.input_proj.bias)
        for proj in [self.alpha_proj, self.beta_proj, self.sigma_proj]:
            nn.init.normal_(proj.weight, 0.0, 0.5)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)   # (B, 80, T)
        h = self.blocks(h)        # (B, 80, T)

        f_alpha = self.alpha_proj(h)   # (B, N,   T)
        f_beta  = self.beta_proj(h)    # (B, N*K, T)
        f_sigma = self.sigma_proj(h)   # (B, N,   T)

        return f_alpha, f_beta, f_sigma

class DiscriminatorTCN(nn.Module):
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int   = 160,
        num_blocks:      int   = 6,
        kernel_size:     int   = 2,
        dilation_base:   int   = 2,
        dropout:         float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            dilation = dilation_base ** i
            blocks.append(
                TemporalBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    dropout      = dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, 0.0, 0.5)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, 0.0, 0.5)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)     # (B, 160, T)
        h = self.blocks(h)          # (B, 160, T)
        h = self.output_proj(h)     # (B, 1,   T)
        score = h.mean(dim=-1).squeeze(1)  # (B,)  — average-pool over time
        return score

class ResidualGeneratorTCN(nn.Module):
    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        hidden_channels: int   = 80,
        num_layers:      int   = 6,
        dropout:         float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    # Verify RFS = 127 for k=2, D=2, L=6
    rfs = TCN.receptive_field_size(kernel_size=2, dilation_base=2, num_blocks=6)
    print(f"Receptive Field Size: {rfs}")   # expected: 127

    # test
    B, T = 4, 200
    N, K = 30, 3 
    dz, dy = 10, 8

    # Generator
    gen = GeneratorTCN(
        in_channels = dz + dy,
        out_alpha   = N,
        out_beta    = N * K,
        out_sigma   = N,
    )
    x_gen = torch.randn(B, dz + dy, T)
    fa, fb, fs = gen(x_gen)
    print(f"Generator outputs — α: {fa.shape}  β: {fb.shape}  σ: {fs.shape}")
    # expected: (4, 98, 200)  (4, 490, 200)  (4, 98, 200)

    # Discriminator
    disc = DiscriminatorTCN(in_channels=N + dy)
    x_disc = torch.randn(B, N + dy, T)
    score = disc(x_disc)
    print(f"Discriminator score shape: {score.shape}")

    # Residual generator
    res_gen = ResidualGeneratorTCN(in_channels=dz + dy, out_channels=N)
    x_res = torch.randn(B, dz + dy, T)
    eps = res_gen(x_res)
    print(f"Residual generator ε shape: {eps.shape}")  

    # Parameter counts
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter counts:")
    print(f"  GeneratorTCN     : {count_params(gen):,}")
    print(f"  DiscriminatorTCN : {count_params(disc):,}")
    print(f"  ResidualGenTCN   : {count_params(res_gen):,}")