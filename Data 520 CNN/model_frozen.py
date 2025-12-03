import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class DualStreamRiceModel(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(DualStreamRiceModel, self).__init__()
        
        print(f"🔒 Backbone Freeze Status: {freeze_backbone}")
        
        # --- Stream A (Radar) ---
        self.backbone_s1 = models.efficientnet_b0(weights='DEFAULT')
        
        if freeze_backbone:
            for param in self.backbone_s1.parameters():
                param.requires_grad = False
                
        self.feature_dim = self.backbone_s1.classifier[1].in_features
        self.backbone_s1.classifier = nn.Identity()
        self.att_s1 = ChannelAttention(1280)

        # --- Stream B (Optical) ---
        self.backbone_s2 = models.efficientnet_b0(weights='DEFAULT')
        
        if freeze_backbone:
            for param in self.backbone_s2.parameters():
                param.requires_grad = False
                
        self.backbone_s2.classifier = nn.Identity()
        self.att_s2 = ChannelAttention(1280)

        # --- Fusion ---
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, s1, s2):
        f1 = self.backbone_s1(s1)
        f1_att = f1.unsqueeze(-1).unsqueeze(-1)
        f1 = f1 * self.att_s1(f1_att).squeeze()

        f2 = self.backbone_s2(s2)
        f2_att = f2.unsqueeze(-1).unsqueeze(-1)
        f2 = f2 * self.att_s2(f2_att).squeeze()

        combined = torch.cat((f1, f2), dim=1)
        output = self.fusion_fc(combined)
        return output