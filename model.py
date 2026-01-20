import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        if x.size(2) != x3.size(2): x = F.interpolate(x, size=(x3.size(2), x3.size(3)))
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        if x.size(2) != x2.size(2): x = F.interpolate(x, size=(x2.size(2), x2.size(3)))
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        if x.size(2) != x1.size(2): x = F.interpolate(x, size=(x1.size(2), x1.size(3)))
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)
        return self.outc(x)


class ResidualBlock(nn.Module):
    """
    A Residual Block with two 3x3 convolutions and a skip connection.
    This allows the model to learn deeper features without gradient issues.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut to match dimensions if input and output differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # The "Residual" connection
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(ResUNet, self).__init__()
        
        # --- ENCODER (Contracting Path) ---
        # Input: 3 x 256 x 256
        self.inc = ResidualBlock(n_channels, 64)
        
        # Down 1: 64 -> 128 (128x128)
        self.down1 = ResidualBlock(64, 128, stride=2)
        
        # Down 2: 128 -> 256 (64x64)
        self.down2 = ResidualBlock(128, 256, stride=2)
        
        # Down 3: 256 -> 512 (32x32)
        self.down3 = ResidualBlock(256, 512, stride=2)
        
        # Bridge (Bottleneck): 512 -> 1024 (16x16)
        self.bridge = ResidualBlock(512, 1024, stride=2)
        
        # --- DECODER (Expansive Path) ---
        # Up 1: 1024 -> 512
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.res_up1 = ResidualBlock(1024, 512) # 1024 because of concat
        
        # Up 2: 512 -> 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.res_up2 = ResidualBlock(512, 256)
        
        # Up 3: 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.res_up3 = ResidualBlock(256, 128)
        
        # Up 4 (Final Resolution Recovery): 128 -> 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res_up4 = ResidualBlock(128, 64)
        
        # Output Layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bridge
        x_bridge = self.bridge(x4)
        
        # Decoder
        # Block 1
        d1 = self.up1(x_bridge)
        # Handle potential rounding errors in size
        if d1.size(2) != x4.size(2): d1 = F.interpolate(d1, size=x4.shape[2:])
        d1 = torch.cat([x4, d1], dim=1) # Skip Connection
        d1 = self.res_up1(d1)
        
        # Block 2
        d2 = self.up2(d1)
        if d2.size(2) != x3.size(2): d2 = F.interpolate(d2, size=x3.shape[2:])
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.res_up2(d2)
        
        # Block 3
        d3 = self.up3(d2)
        if d3.size(2) != x2.size(2): d3 = F.interpolate(d3, size=x2.shape[2:])
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.res_up3(d3)
        
        # Block 4 (New block to get back to 256x256)
        d4 = self.up4(d3)
        if d4.size(2) != x1.size(2): d4 = F.interpolate(d4, size=x1.shape[2:])
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.res_up4(d4)
        
        return self.outc(d4)