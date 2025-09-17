import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
class Encoder(nn.Module):
    """
      Conv1: 1->32 + MaxPool (128->64)
      Conv2: 32->64 + MaxPool (64->32)
      Conv3: 64->128 + MaxPool (32->16)
      Conv4: 128->256 + MaxPool (16->8)
    """
    def __init__(self):
        super().__init__()
        self.c1 = conv_block(1, 32)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_block(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_block(128, 256)
        self.p4 = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.c1(x)        
        x = self.p1(f1)         
        f2 = self.c2(x)         
        x = self.p2(f2)        
        f3 = self.c3(x)        
        x = self.p3(f3)        
        f4 = self.c4(x)        
        x = self.p4(f4)  
        return f1, f2, f3, f4, x 


class HeatmapNet(nn.Module):
    """
    U-Net-like decoder to produce [B, K, 64, 64] heatmaps from 128x128  input.
    """
    def __init__(self, num_keypoints: int = 5):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.enc = Encoder()
        # Decoder:
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 8->16
        self.dc4 = conv_block(384, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 16->32
        self.dc3 = conv_block(192, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 32->64
        self.dc2 = conv_block(96, 32)
        self.out_conv = nn.Conv2d(32, self.num_keypoints, kernel_size=1)

    def forward(self, x):
      f1, f2, f3, f4, bottleneck = self.enc(x)

      x = self.up4(bottleneck)
      if x.shape[-2:] != f4.shape[-2:]:
        x = F.interpolate(x, size=f4.shape[-2:], mode="bilinear", align_corners=False)
      x = torch.cat([x, f4], dim=1)
      x = self.dc4(x)
      
      x = self.up3(x)
      if x.shape[-2:] != f3.shape[-2:]:
        x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
      x = torch.cat([x, f3], dim=1)
      x = self.dc3(x)

      x = self.up2(x)
      if x.shape[-2:] != f2.shape[-2:]:
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
      x = torch.cat([x, f2], dim=1)
      x = self.dc2(x)

      heatmaps = self.out_conv(x)   # shape [B, K, 64, 64]
      return heatmaps

class RegressionNet(nn.Module):
    def __init__(self, num_keypoints: int = 5):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.enc = Encoder()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_keypoints * 2)

    def forward(self, x):
        _, _, _, _, bottleneck = self.enc(x)
        x = self.gap(bottleneck).flatten(1)   
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        coords = torch.sigmoid(self.fc3(x))  
        return coords