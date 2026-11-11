"""Parametric pixel/history actor. No lookup, RAM, level or clock inputs."""
import torch
from torch import nn
from torch.nn import functional as F

HISTORY_LENGTH = 128


class RobustPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shifts', torch.arange(10), persistent=False)
        self.pixels = nn.Sequential(
            nn.Conv2d(12, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512), nn.ReLU())
        self.history = nn.Sequential(nn.Linear(HISTORY_LENGTH * 11, 256), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(512 + 256 + 4, 512), nn.ReLU(), nn.Dropout(.1))
        self.actor = nn.Linear(512, 10)
        self.value = nn.Linear(512, 1)

    def forward(self, frames, past, valid=None):
        if frames.dtype != torch.uint8 or frames.ndim != 5 or tuple(frames.shape[1:]) != (4, 128, 128, 3) or frames.shape[0] < 1:
            raise ValueError('Expected nonempty uint8 frames[B,4,128,128,3]')
        batch = frames.shape[0]
        if past.dtype != torch.int64 or past.shape != (batch, HISTORY_LENGTH) or past.device != frames.device:
            raise ValueError('Expected int64 past[B,128] on frame device')
        if torch.any((past < -1) | (past > 1023)):
            raise ValueError('Past requires ten-bit masks or -1 for genuinely unavailable history')
        if valid is None:
            valid = torch.ones((batch, 4), dtype=torch.bool, device=frames.device)
        if valid.dtype != torch.bool or valid.shape != (batch, 4) or valid.device != frames.device:
            raise ValueError('Expected boolean valid[B,4] on frame device')
        images = frames.float().mul(1. / 255.) * valid[:, :, None, None, None]
        images = images.permute(0, 1, 4, 2, 3).reshape(batch, 12, 128, 128)
        bits = ((past.clamp_min(0).unsqueeze(-1) >> self.shifts) & 1).float()
        actions = torch.cat((bits, (past >= 0).unsqueeze(-1).float()), -1).reshape(batch, -1)
        hidden = self.trunk(torch.cat((self.pixels(images), self.history(actions), valid.float()), -1))
        # Each bit stays strictly stochastic in FP32; joint probabilities use FP64.
        logits = 12. * torch.tanh(self.actor(hidden) / 12.)
        return logits, self.value(hidden).squeeze(-1)


def joint_log_prob(logits, masks):
    """One full virtual-mask probability; aliases remain distinct labels."""
    if logits.ndim != 2 or logits.shape[1] != 10 or masks.shape != logits.shape[:1] or masks.dtype != torch.int64:
        raise ValueError('Expected logits[B,10] and int64 masks[B]')
    if torch.any((masks < 0) | (masks > 1023)):
        raise ValueError('Invalid virtual mask')
    bits = (masks[:, None] >> torch.arange(10, device=masks.device)) & 1
    signed = logits.double() * (2 * bits - 1)
    return F.logsigmoid(signed).sum(-1)


def joint_entropy(logits):
    value = logits.double()
    return (F.softplus(value) - torch.sigmoid(value) * value).sum(-1)
