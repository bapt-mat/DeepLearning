import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Contracting Path (Encoder) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- Expansive Path (Decoder) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512) # 1024 because of concat (512+512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        # Handle skip connection size mismatch (if any)
        if x.size(2) != x4.size(2): x = F.interpolate(x, size=x4.shape[2:])
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        if x.size(2) != x3.size(2): x = F.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        if x.size(2) != x2.size(2): x = F.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        if x.size(2) != x1.size(2): x = F.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits