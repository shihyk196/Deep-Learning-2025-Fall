import torch
import torch.nn as nn
import torch.nn.functional as F

class LDB(nn.Module):
    def __init__(self, in_channel: int, t: float = 0.5):
        super(LDB, self).__init__()
        mid_channel = int(in_channel * t)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 1x1 conv 壓縮
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channel)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 conv 分支 1
        self.bn2   = nn.BatchNorm2d(mid_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)

        # 3x3 conv 分支 2
        self.bn3   = nn.BatchNorm2d(mid_channel)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)

        # 輸出通道數 = in_channel + 4 * mid_channel (依你的設計)
        # self.out_channels = in_channel + 4 * mid_channel
        self.out_channels = in_channel
        # 把 mid_channel 再拉回 in_channel
        self.conv_out = nn.Conv2d(mid_channel, in_channel, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out1 = self.conv2(self.relu2(self.bn2(out)))
        out2 = self.conv3(self.relu3(self.bn3(out)))

        fused = out1 + out2
        fused = self.conv_out(fused)  # 把 mid_channel 拉回 in_channel

        return x + self.alpha * fused              # 殘差連接

# ------------------------------
# Transition Layer
# ------------------------------
class Transition(nn.Module):
    def __init__(self, in_channel: int, out_channel: int = 32):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = out_channel

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CDenseNet(nn.Module):
    def __init__(self, n: int = 16, t: float = 0.5, num_outputs: int = 3):
        """
        n: LDB block 數量
        t: channel scaling factor
        num_outputs: crowd counting 的輸出維度 (UCSD = 3: away, toward, total)
        """
        super(CDenseNet, self).__init__()

        # Stem (輸入灰階圖像 1 channel)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        channels = 32
        blocks = []
        for _ in range(n):
            ldb = LDB(channels, t)
            channels = ldb.out_channels
            trans = Transition(channels, out_channel=32)
            channels = trans.out_channels
            blocks.append(ldb)
            blocks.append(trans)

        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_outputs)  # [away, toward, total]


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)   
        x = self.fc2(x)
        return x

# if __name__ == "__main__":
#     model = CDenseNet(n=16, t=0.5, num_outputs=3)
#     dummy = torch.randn(1, 1, 158, 238)  # UCSD: 單通道灰階 158x238
#     out = model(dummy)
#     print("Output shape:", out.shape)  # 預期 [1, 3]