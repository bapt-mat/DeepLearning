import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# --- CUSTOM BLOCKS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

# --- DEEP SUPERVISION ARCHITECTURE ---
class DeepSupResUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # Encoder
        self.inc = ResidualBlock(3, 64)
        self.d1 = ResidualBlock(64, 128, 2)
        self.d2 = ResidualBlock(128, 256, 2)
        self.d3 = ResidualBlock(256, 512, 2)
        self.bridge = ResidualBlock(512, 1024, 2)
        # Decoder
        self.u1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.ru1 = ResidualBlock(1024, 512)
        self.u2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.ru2 = ResidualBlock(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.ru3 = ResidualBlock(256, 128)
        self.u4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.ru4 = ResidualBlock(128, 64)
        # Heads
        self.out_final = nn.Conv2d(64, n_classes, 1)      # 256x256
        self.out_ds1 = nn.Conv2d(128, n_classes, 1)       # 128x128
        self.out_ds2 = nn.Conv2d(256, n_classes, 1)       # 64x64

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        b = self.bridge(x4)
        d1 = self.ru1(torch.cat([x4, self.u1(b)], 1))
        d2 = self.ru2(torch.cat([x3, self.u2(d1)], 1))
        out2 = self.out_ds2(d2) # Side Output
        d3 = self.ru3(torch.cat([x2, self.u3(d2)], 1))
        out1 = self.out_ds1(d3) # Side Output
        d4 = self.ru4(torch.cat([x1, self.u4(d3)], 1))
        out0 = self.out_final(d4) # Final Output
        
        if self.training: return [out0, out1, out2]
        else: return out0

# --- FLEXIBLE WRAPPER ---
class FlexibleModel(nn.Module):
    def __init__(self, arch='unet', encoder='resnet34', weights='imagenet', n_classes=1):
        super(FlexibleModel, self).__init__()
        self.arch = arch
        if arch == 'deepsup':
            self.model = DeepSupResUNet(n_classes=n_classes)
        elif arch == 'unet':
            self.model = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=n_classes)
        elif arch == 'segformer':
            self.model = smp.Segformer(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=n_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x):
        return self.model(x)