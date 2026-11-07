"""Detector evidence tokens: every candidate of the detector branch enters the language model as ONE learned token
instead of a line of text (about 25 tokens), so the prompt of a covered question shrinks to the question itself plus
at most eight tokens. On an edge NPU, where prefill time grows with the number of input tokens, this is the cheapest
form of the evidence interface.

  token of a candidate = class embedding + MLP(Fourier features of confidence and box)
  class embedding   initialised with the mean input embedding of the class name, so that at the start of training the
                    token already "means" the class to the language model; trainable
  MLP               last layer initialised to zero: confidence and position are learned as a correction

The language model sees the tokens at the positions of a placeholder special token (core.EVIDENCE_SLOT).
"""
from __future__ import annotations

import math

import torch
from torch import nn

import bench

CLASSES = list(bench.ED_NAMES) + list(bench.TD_CLASSES)  # one table for both scenarios


def features(given, width, height):
    """(class indices [K], features [K, 5]) of a record written by `verify.py evidence`: confidence and the box
    corners as fractions of the image."""
    dets = given["dets"]
    cls = torch.tensor([CLASSES.index(d["cls"]) for d in dets], dtype=torch.long)
    feat = torch.tensor([[d["conf"], d["box"][0] / width, d["box"][1] / height, d["box"][2] / width, d["box"][3] / height] for d in dets], dtype=torch.float32).reshape(-1, 5)
    return cls, feat.clamp(0.0, 1.0)


class EvidenceProjector(nn.Module):
    def __init__(self, dim, hidden=512, bands=8):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(len(CLASSES), dim))
        self.mlp = nn.Sequential(nn.Linear(5 * 2 * bands, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("freq", (2.0 ** torch.arange(bands)) * math.pi, persistent=False)

    @torch.no_grad()
    def init_from_names(self, embed_table, tokenizer):
        """Class embeddings start as the mean input embedding of the class's display name."""
        for k, c in enumerate(CLASSES):
            ids = tokenizer.encode(bench.ED_NAMES.get(c, c), add_special_tokens=False)
            self.cls[k] = embed_table[torch.tensor(ids, device=embed_table.device)].float().mean(0)

    def forward(self, cls, feat):
        x = feat[..., None] * self.freq  # [K, 5, bands]
        x = torch.cat([x.sin(), x.cos()], dim=-1).flatten(1)
        return self.cls[cls] + self.mlp(x)
