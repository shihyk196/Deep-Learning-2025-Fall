import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
#  強化版：base_c 維持 8，不惜一切代價撐容量
# ============================================
NUM_CLASSES = 8
DEFAULT_BASE_C = 8   # <<< 保持不動

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        min_channel = 4
        reduced = max(channel // reduction, min_channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 expand_ratio=1.5, use_se=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_se = use_se
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ))
        else:
            hidden_dim = in_channels

        # dw
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        if self.use_se:
            layers.append(SEBlock(hidden_dim))

        # pw-linear
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class NanoASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NanoASPP, self).__init__()
        # 中間通道數：變胖（原本是 in_channels // 2）
        mid = in_channels * 3 // 4   # 例如 in_channels=32 -> mid=24

        # 分支 1：1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # 分支 2：dilated depthwise 3x3, rate=2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=2, dilation=2,
                groups=in_channels,  # depthwise
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # 分支 3：dilated depthwise 3x3, rate=4
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=4, dilation=4,
                groups=in_channels,  # depthwise
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # 分支 4：global pooling
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # 最後再融合 4 個分支：4 * mid -> out_channels
        self.project = nn.Sequential(
            nn.Conv2d(mid * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.global_avg(x)
        b4 = F.interpolate(b4, size=b1.shape[2:], mode='nearest')

        x = torch.cat([b1, b2, b3, b4], dim=1)
        return self.project(x)

class MobileLiteUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=NUM_CLASSES,
                 base_c=DEFAULT_BASE_C):
        super(MobileLiteUNet, self).__init__()

        # Initial Conv [H/2, W/2]
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, base_c, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(base_c),
            nn.ReLU6(inplace=True)
        )

        # ============ Encoder（加深 / 加強） ============
        # L1: [H/4, W/4] (stride=2) -> 2 * base_c
        self.layer1 = InvertedResidual(
            base_c, base_c * 2,
            stride=2, expand_ratio=2.0, use_se=True
        )

        # L2: [H/8, W/8] (stride=2) -> 3 * base_c
        self.layer2 = nn.Sequential(
            InvertedResidual(
                base_c * 2, base_c * 3,
                stride=2, expand_ratio=2.0, use_se=True
            ),
            InvertedResidual(
                base_c * 3, base_c * 3,
                stride=1, expand_ratio=2.0, use_se=True
            ),
        )

        # L3: [H/16, W/16] (stride=2) -> 4 * base_c
        # 多一層 block，bottleneck 更深
        self.layer3 = nn.Sequential(
            InvertedResidual(
                base_c * 3, base_c * 4,
                stride=2, expand_ratio=2.5, use_se=True
            ),
            InvertedResidual(
                base_c * 4, base_c * 4,
                stride=1, expand_ratio=2.5, use_se=True
            ),
            InvertedResidual(
                base_c * 4, base_c * 4,
                stride=1, expand_ratio=2.5, use_se=True
            ),
        )

        # Bottleneck ASPP
        self.aspp = NanoASPP(base_c * 4, base_c * 4)

        # ============ Decoder（開 SE + 撐 expand_ratio） ============
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        # In: 4c(ASPP) + 3c(L2) = 7c -> Out: 3c
        self.dec3 = InvertedResidual(
            base_c * 7, base_c * 3,
            stride=1, expand_ratio=1.5, use_se=True
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        # In: 3c(D3) + 2c(L1) = 5c -> Out: 2c
        self.dec2 = InvertedResidual(
            base_c * 5, base_c * 2,
            stride=1, expand_ratio=1.5, use_se=True
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # In: 2c(D2) + 1c(Inc) = 3c -> Out: 2c
        self.dec1 = InvertedResidual(
            base_c * 3, base_c * 2,
            stride=1, expand_ratio=1.5, use_se=True
        )

        # Final upsample & heads
        self.final_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.outc = nn.Conv2d(base_c * 2, n_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(base_c * 2, n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)     # H/2
        x1 = self.layer1(x0) # H/4
        x2 = self.layer2(x1) # H/8
        x3 = self.layer3(x2) # H/16

        x_center = self.aspp(x3)

        d3 = self.up3(x_center)  # H/8
        if d3.size()[2:] != x2.size()[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='nearest')
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)       # H/4
        if d2.size()[2:] != x1.size()[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='nearest')
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)      # 2c

        # Deep supervision branch：只在 training 時算
        aux_logits = None
        if self.training:
            aux_logits = self.aux_head(d2)  # (B, num_classes, H/4, W/4)
            aux_logits = F.interpolate(
                aux_logits,
                size=x.shape[2:],          # 原始輸入大小 H, W
                mode='bilinear',
                align_corners=False
            )

        d1 = self.up1(d2)       # H/2
        if d1.size()[2:] != x0.size()[2:]:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode='nearest')
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out = self.final_up(d1) # H
        logits = self.outc(out) # (B, num_classes, H, W)

        # 訓練：回傳 (main, aux)；推論：只回 main
        if self.training:
            return logits, aux_logits
        else:
            return logits

def load_model(MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = MobileLiteUNet(
        n_channels=3,
        n_classes=NUM_CLASSES,
        base_c=DEFAULT_BASE_C
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
