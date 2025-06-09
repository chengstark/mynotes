from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio.transforms as T

from template_base import Template, GeneSpec, register_template
from nas_ga import NAS

CHANNEL_OPTS = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 128, 156, 256]
KERNEL_OPTS  = [3, 5, 7, 9]
DROPOUT_OPTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

class MFCCBlock(nn.Module):
    """Faithful re-implementation of the user’s original `MFCC_block`.

    * Produces a 4-D tensor shaped **(B, 10, T, 1)** – the `10` MFCC features
      live in the *channel* dimension, matching the first conv layer.
    * Keeps optional batch-norm over those channels.
    """
    def __init__(self, bnorm_outputs: bool = True, *, sr: int = 16000, window: int = 40,
                 stride: int = 20, n_mels: int = 128, feats_per_window: int = 10):
        super().__init__()
        self.n_mfcc = feats_per_window
        n_fft = int(sr * window / 1000)
        hop = int(sr * stride / 1000)
        self.mfcc = T.MFCC(sample_rate=sr, n_mfcc=self.n_mfcc,
                           melkwargs=dict(n_fft=n_fft, n_mels=n_mels, hop_length=hop,
                                          mel_scale="htk", center=False))
        self.bn = nn.BatchNorm2d(feats_per_window) if bnorm_outputs else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, N)
        with torch.no_grad():
            x = self.mfcc(x)          # (B, n_mfcc, T)
            x = torch.transpose(x, 2, 3)
        x = x.permute(0, 3, 2, 1)
        x = self.bn(x)  # back to (B, n_mfcc, T, 1)
        return x

class DepthwiseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=2):
        super().__init__(); pad = (k - 1) // 2
        self.short = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)), nn.BatchNorm2d(out_ch))
        
        def dw(c, s=1): 
            return nn.Conv2d(c, c, (k, 1), padding=(pad, 0), stride=(s, 1), groups=c)
        
        def pw(ci, co): 
            return nn.Conv2d(ci, co, 1)
        
        self.seq = nn.Sequential(
            dw(in_ch, stride), 
            nn.BatchNorm2d(in_ch), 
            nn.ReLU(),
            pw(in_ch, out_ch), 
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            dw(out_ch), 
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            pw(out_ch, out_ch), 
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.seq(x) + self.short(x))

@register_template
class TCDSResNet14Template(Template):
    GENE_SCHEMA = (
        GeneSpec("stem_channels", CHANNEL_OPTS),
        GeneSpec("dropout", DROPOUT_OPTS),
    )
    BUNDLE_TYPES = {
        0: ("conv", (GeneSpec("ch", CHANNEL_OPTS), GeneSpec("k", KERNEL_OPTS))),
    }
    MAX_BUNDLES = 20

    def __init__(self, num_classes=12):
        super().__init__(); self.num_classes = num_classes

    def build(self, g):
        stem_ch = CHANNEL_OPTS[g["stem_channels"]]
        drop    = DROPOUT_OPTS[g["dropout"]]
        in_ch = stem_ch
        layers = [
            MFCCBlock(),
            nn.Conv2d(10, in_ch, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(in_ch), nn.ReLU()
        ]
        for tag, params in g["bundles"]:
            ch_idx, k_idx = params
            out_ch = CHANNEL_OPTS[ch_idx]
            k = KERNEL_OPTS[k_idx]
            layers.append(DepthwiseBlock(in_ch, out_ch, k, stride=2))
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(in_ch, self.num_classes)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    NUM_CLASSES = 12
    SAMPLE_LEN = 16000

    def fake_batch(batch_size: int = 8):
        x = torch.randn(batch_size, 1, SAMPLE_LEN)
        y = torch.randint(0, NUM_CLASSES, (batch_size,))
        return x, y

    def train_fn(model: nn.Module, steps: int = 2) -> float:
        model.train(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(steps):
            x, y = fake_batch(); opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
        with torch.no_grad():
            model.eval(); x, y = fake_batch(); acc = (model(x).argmax(1) == y).float().mean().item()
        return acc

    nas = NAS(TCDSResNet14Template, num_classes=NUM_CLASSES)
    nas.evolve(population=6, generations=2, train_fn=train_fn)
    print("Best fitness:", nas.best_score_)
