import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_c // 2 + skip_c, out_c),
            ConvBnRelu(out_c, out_c),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            dy = skip.size(2) - x.size(2)
            dx = skip.size(3) - x.size(3)
            x  = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            x  = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MobileUNet(nn.Module):
    def __init__(self, n_classes=2, freeze_encoder=True):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features

        self.enc0 = backbone[0:2]
        self.enc1 = backbone[2:4]
        self.enc2 = backbone[4:7]
        self.enc3 = backbone[7:14]
        self.enc4 = backbone[14:19]

        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

        self.bottleneck = nn.Sequential(ConvBnRelu(1280, 256), ConvBnRelu(256, 256))
        self.dec4 = DecoderBlock(256,  96, 128)
        self.dec3 = DecoderBlock(128,  32,  64)
        self.dec2 = DecoderBlock( 64,  24,  32)
        self.dec1 = DecoderBlock( 32,  16,  32)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ConvBnRelu(32, 32),
        )
        self.head = nn.Conv2d(32, n_classes, kernel_size=1)

        for m in [self.bottleneck, self.dec4, self.dec3, self.dec2, self.dec1, self.dec0, self.head]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x  = self.bottleneck(s4)
        x  = self.dec4(x, s3)
        x  = self.dec3(x, s2)
        x  = self.dec2(x, s1)
        x  = self.dec1(x, s0)
        x  = self.dec0(x)
        return self.head(x)
