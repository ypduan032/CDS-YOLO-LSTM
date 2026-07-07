import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):

    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
        )

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        lengths = torch.clamp(torch.div(lengths + 1, 2, rounding_mode="trunc"), min=1)
        return x, lengths


class ChannelAttention1D(nn.Module):
    """CBAM channel attention for 1D sequences."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.to(x.device)
            mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
            mask_f = mask.float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            avg_pool = (x * mask_f).sum(dim=1) / denom
            max_pool = x.masked_fill(~mask, -1e9).max(dim=1).values
        else:
            avg_pool = x.mean(dim=1)
            max_pool = x.max(dim=1).values
        gate = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return gate.unsqueeze(1)


class BiLevelRoutingAttention1D(nn.Module):
    """BRA-style window routing for 1D sequences."""

    def __init__(self, channels, window_size=8, num_routes=8, topk=4):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_routes = num_routes
        self.topk = topk
        self.norm = nn.LayerNorm(channels)
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.region_proj = nn.Linear(channels, channels)
        hidden = max(channels // 2, 1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def _pad_to_window(self, x):
        b, t, c = x.shape
        pad_len = (self.window_size - (t % self.window_size)) % self.window_size
        if pad_len:
            x = torch.cat([x, x.new_zeros(b, pad_len, c)], dim=1)
        return x, t

    def forward(self, x, lengths=None):
        x_norm = self.norm(x)
        x_pad, t_orig = self._pad_to_window(x_norm)
        local = self.dwconv(x_pad.transpose(1, 2)).transpose(1, 2)

        q = self.q_proj(x_pad)
        k = self.k_proj(x_pad)
        v = self.v_proj(x_pad)

        b, t_pad, c = x_pad.shape
        nwin = t_pad // self.window_size
        q_win = q.view(b, nwin, self.window_size, c)
        k_win = k.view(b, nwin, self.window_size, c)
        v_win = v.view(b, nwin, self.window_size, c)

        region_k = self.region_proj(k_win.mean(dim=2))
        region_v = self.region_proj(v_win.mean(dim=2))
        if region_k.size(1) > self.num_routes:
            region_k = F.adaptive_avg_pool1d(region_k.transpose(1, 2), self.num_routes).transpose(1, 2)
            region_v = F.adaptive_avg_pool1d(region_v.transpose(1, 2), self.num_routes).transpose(1, 2)

        token_q = q_win.reshape(b, nwin * self.window_size, c)
        scores = torch.matmul(token_q, region_k.transpose(1, 2)) / math.sqrt(self.channels)
        topk = min(self.topk, scores.size(-1))
        topk_scores, topk_idx = torch.topk(scores, k=topk, dim=-1)
        masked_scores = torch.full_like(scores, float("-inf"))
        masked_scores.scatter_(-1, topk_idx, topk_scores)
        routing_weights = torch.softmax(masked_scores, dim=-1)
        routing_context = torch.matmul(routing_weights, region_v)
        routing_context = routing_context.view(b, nwin, self.window_size, c).reshape(b, t_pad, c)

        enhanced = local + routing_context
        gate = torch.sigmoid(self.gate(enhanced))[:, :t_orig, :]
        if lengths is not None:
            lengths = lengths.to(x.device)
            mask = (torch.arange(t_orig, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
            gate = gate * mask.float()
        return gate


class CBAMBRAAttention(nn.Module):
    """Channel attention + Bi-Level Routing Attention."""

    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelAttention1D(channels)
        self.bra_attn = BiLevelRoutingAttention1D(channels)

    def forward(self, x, lengths=None):
        x = x * self.channel_attn(x, lengths)
        x = x * self.bra_attn(x, lengths)
        return x


class ImprovedBiLSTM(nn.Module):
    """CNN -> BiLSTM -> CBAM-BRA -> Output."""

    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels=128, out_channels=128)
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = CBAMBRAAttention(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    @staticmethod
    def masked_mean(x, lengths):
        lengths = lengths.to(x.device)
        mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        mask_f = mask.float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    def forward(self, x, lengths):
        x, lengths = self.cnn(x, lengths)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.attn(out, lengths)
        context = self.masked_mean(out, lengths)
        return self.fc(context).squeeze(1)
