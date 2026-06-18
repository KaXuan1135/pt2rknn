import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3_1DTCN(nn.Module):
    def __init__(self, num_classes=8, num_frames=16, hidden_dim=576):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # 1D-TCN temporal classifier
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)
        self.num_frames = num_frames

    def forward(self, x):
        feat = self.backbone(x)           # [T, 576, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)  # [T, 576]
        feat = feat.unsqueeze(0)          # [1, 576, T] after permute
        feat = feat.permute(0, 2, 1)      # [1, 576, T]
        # TCN
        feat = self.tcn(feat)             # [1, 128, T]
        feat = feat.mean(dim=-1)          # [1, 128] global avg pooling
        out = self.classifier(feat)       # [1, num_classes]
        return out