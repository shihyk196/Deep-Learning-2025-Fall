import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
#  Simple U-Net for Semantic Segmentation
#  Input:  3 x H x W (RGB image)
#  Output: 8 x H x W (logits for each class)
# ============================================

NUM_CLASSES = 8   # 這裡改成 8 類

class DoubleConv(nn.Module):
    """
    Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling: MaxPool2d(2) -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling: bilinear upsampling + concat(skip) + DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if use bilinear up, only need a 1x1 conv to reduce channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # transposed conv
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: deeper feature map
        # x2: skip connection from encoder
        x1 = self.up(x1)

        # 尺寸對齊（理論上 256x512 經 2^n 下採樣不會有問題，這裡保險一點做對齊）
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # concat along channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 conv to get class logits
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 主體
    """
    def __init__(self, n_channels=3, n_classes=NUM_CLASSES, base_c=64, bilinear=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc   = DoubleConv(n_channels, base_c)          # 3 -> 64
        self.down1 = Down(base_c, base_c * 2)                # 64 -> 128
        self.down2 = Down(base_c * 2, base_c * 4)            # 128 -> 256
        self.down3 = Down(base_c * 4, base_c * 8)            # 256 -> 512
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor) # 512 -> 1024 (or 512 if bilinear)

        # Decoder
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear=bilinear)

        # Output
        self.outc = OutConv(base_c, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # 3 -> 64, HxW
        x2 = self.down1(x1)  # 64 -> 128, H/2 x W/2
        x3 = self.down2(x2)  # 128 -> 256, H/4 x W/4
        x4 = self.down3(x3)  # 256 -> 512, H/8 x W/8
        x5 = self.down4(x4)  # 512 -> 1024, H/16 x W/16

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)  # B x NUM_CLASSES x H x W
        return logits


# ============================================
#  load_model: 給 test_task1.py 呼叫
# ============================================

def load_model(MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立和訓練時一樣的模型結構
    model = UNet(n_channels=3, n_classes=NUM_CLASSES)

    # 把權重載到對應 device（通常就是 cuda）
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # 模型本體也搬到 device
    model = model.to(device)

    model.eval()
    return model
