"""EchoNet: two-branch CNN front end + selectable temporal model.

    doppler [B,K,F,T] --BN--> 2D residual CNN (pools frequency, T/4) --\
                                                                         concat -> 1x1 -> temporal -> attention pool -> classes
    iq      [B,M,T]   --BN--> 1D residual CNN (T/4)  ------------------/

``temporal``:
    none        CNN only (the plan's baseline)
    gru         2-layer bidirectional GRU (CNN-GRU)
    transformer pre-norm Transformer encoder with learned positions (CNN-Transformer)
    tcn         dilated residual temporal convolutions
"""

from __future__ import annotations

import torch
from torch import nn

from .config import Config
from .features import doppler_bins, frames_per_window, iq_channels


class Res2d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = (nn.Identity() if cin == cout and stride == (1, 1) else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class Res1d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dilation: int = 1, k: int = 5):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout), nn.GELU(),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation, bias=False), nn.BatchNorm1d(cout),
        )
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class DopplerCNN(nn.Module):
    def __init__(self, tones: int, width: int):
        super().__init__()
        w2 = int(width * 1.5)
        # Downsample time early (lip/jaw motion is < 15 Hz) so the CNN stays cheap on CPU.
        self.net = nn.Sequential(
            nn.BatchNorm2d(tones),
            nn.Conv2d(tones, width // 2, 3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(width // 2), nn.GELU(),
            Res2d(width // 2, width, (2, 2)),
            Res2d(width, w2, (2, 1)),
        )
        self.out_channels = w2

    def forward(self, x):  # [B,K,F,T] -> [B,C,T/4]
        return self.net(x).mean(dim=2)


class IQCNN(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        w2 = int(width * 1.5)
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, width, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(width), nn.GELU(),
            Res1d(width, width),
            Res1d(width, w2, stride=2),
        )
        self.out_channels = w2

    def forward(self, x):  # [B,M,T] -> [B,C,T/4]
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, h):  # [B,T,D] -> [B,2D]: attention mean and max
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return torch.cat([(w.unsqueeze(-1) * h).sum(dim=1), h.amax(dim=1)], dim=-1)


class EchoNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m, f = cfg.model, cfg.features
        k = len(cfg.probe.tones_hz)
        self.branches = m.branches
        self.temporal_kind = m.temporal
        self.doppler = DopplerCNN(k, m.channels) if m.branches in ("both", "doppler") else None
        self.iq = IQCNN(iq_channels(k, f), m.channels) if m.branches in ("both", "iq") else None
        cin = (self.doppler.out_channels if self.doppler else 0) + (self.iq.out_channels if self.iq else 0)
        d = m.d_model
        self.proj = nn.Sequential(nn.Conv1d(cin, d, 1), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(m.dropout * 0.5))
        tokens = frames_per_window(f, cfg.data.window_s) // 4 + 2
        if m.temporal == "gru":
            self.temporal = nn.GRU(d, d // 2, num_layers=m.layers, batch_first=True, bidirectional=True,
                                   dropout=m.dropout if m.layers > 1 else 0.0)
        elif m.temporal == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, tokens, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, m.heads, dim_feedforward=2 * d, dropout=m.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, m.layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
        elif m.temporal == "tcn":
            self.temporal = nn.Sequential(*[Res1d(d, d, dilation=2 ** i, k=3) for i in range(m.layers + 1)])
        else:
            self.temporal = None
        self.pool = AttentionPool(d)
        self.head = nn.Sequential(nn.Dropout(m.dropout), nn.Linear(2 * d, len(cfg.classes)))
        self.expected = {"doppler": (k, doppler_bins(f)), "iq": iq_channels(k, f)}

    def forward(self, doppler: torch.Tensor, iq: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.doppler is not None:
            parts.append(self.doppler(doppler))
        if self.iq is not None:
            parts.append(self.iq(iq))
        t = min(p.shape[-1] for p in parts)
        h = self.proj(torch.cat([p[..., :t] for p in parts], dim=1))  # [B,D,T']
        if self.temporal_kind == "tcn":
            h = self.temporal(h)
        h = h.transpose(1, 2)  # [B,T',D]
        if self.temporal_kind == "gru":
            h, _ = self.temporal(h)
        elif self.temporal_kind == "transformer":
            h = self.norm(self.temporal(h + self.pos[:, : h.shape[1]]))
        return self.head(self.pool(h))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
