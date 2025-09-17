import torch.nn as nn

def _cbr(inp, out, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True),
    )

class DetectionHead(nn.Module):
    def __init__(self, in_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.out  = nn.Conv2d(in_ch, num_anchors * (5 + num_classes), 1)
    def forward(self, x): return self.out(self.conv(x))

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):

      super().__init__()
      self.block1 = nn.Sequential(_cbr(3, 32, 1), _cbr(32, 64, 2))  # 224->112
      self.block2 = _cbr(64, 128, 2)   # 112->56
      self.block3 = _cbr(128, 256, 2)  # 56->28
      self.block4 = _cbr(256, 512, 2)  # 28->14

      self.h1 = DetectionHead(128, num_anchors, num_classes)  # 56x56
      self.h2 = DetectionHead(256, num_anchors, num_classes)  # 28x28
      self.h3 = DetectionHead(512, num_anchors, num_classes)  # 14x14

    def forward(self, x):
      x  = self.block1(x)
      s1 = self.block2(x)
      s2 = self.block3(s1)
      s3 = self.block4(s2)
      return [self.h1(s1), self.h2(s2), self.h3(s3)]
