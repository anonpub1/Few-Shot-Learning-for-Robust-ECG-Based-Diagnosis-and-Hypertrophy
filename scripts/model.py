import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): 
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x): 
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, d, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, k, padding=(k-1)*d, dilation=d),
            Chomp1d((k-1)*d), nn.BatchNorm1d(n_out), nn.ReLU(),
            nn.Conv1d(n_out, n_out, k, padding=(k-1)*d, dilation=d),
            Chomp1d((k-1)*d), nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(p)
        )
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.down(x) if self.down else x
        return self.relu(self.net(x) + res)

class TCNEncoder(nn.Module):
    def __init__(self, n_in=3, n_ch=[64, 128, 256], k=7, embed_dim=256):
        super().__init__()
        layers = []
        for i in range(len(n_ch)):
            dilation = 2 ** i
            in_channels = n_in if i == 0 else n_ch[i - 1]
            layers.append(TemporalBlock(in_channels, n_ch[i], k, dilation, p=0.2))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(n_ch[-1], embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return self.fc(x[:, :, -1])
