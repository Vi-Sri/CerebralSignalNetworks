import torch
import torchvision

import torch.nn as nn

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone_image = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone_eeg = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=1, stride=1, padding=1, bias=False),
            torchvision.models.resnet50(zero_init_residual=True),
        )
        self.backbone.fc = nn.Identity()

        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone_image(y1))
        z2 = self.projector(self.backbone_eeg(y2))
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss