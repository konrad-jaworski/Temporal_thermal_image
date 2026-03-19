import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F 


class BnetMean(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=256,
        output_width=512
    ):
        super().__init__()

        # --- U-Net backbone ---
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=decoder_channels,  # feature maps, not final output
            activation=None
        )

        # --- Column-wise regression head ---
        self.regressor = nn.Sequential(
            nn.Conv1d(decoder_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1)
        )

        self.output_width = output_width

    def forward(self, x):
        """
        x: [B, 3, 512, 512]
        return: [B, 512]
        """

        # U-Net output: [B, C, H, W]
        feat = self.unet(x)

        # Pool over height (H)
        feat = feat.mean(dim=2)  # [B, C, W]

        # Regress per column
        out = self.regressor(feat)  # [B, 1, W]

        out = out.squeeze(1)  # [B, W]

        # Optional: enforce [0,1]
        out = torch.sigmoid(out)

        return out

class HierarchicalVerticalProjection(nn.Module):
    """
    Gradually compresses height dimension using
    stacked vertical convolutions with non-linearity.
    """
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(15, 1), stride=(2, 1), padding=(7, 0)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=(15, 1), stride=(2, 1), padding=(7, 0)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=(15, 1), stride=(2, 1), padding=(7, 0)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=(15, 1), stride=(2, 1), padding=(7, 0)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # Final collapse to height = 1
            nn.Conv2d(channels, channels, kernel_size=(32, 1))
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.net(x)       # [B, C, 1, W]
        return x.squeeze(2)  # [B, C, W]

class BnetSmallKernel(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=256
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=decoder_channels,
            activation=None
        )

        self.vertical_proj = HierarchicalVerticalProjection(decoder_channels)

        self.regressor = nn.Sequential(
            nn.Conv1d(decoder_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        feat = self.unet(x)           # [B, C, H, W]
        feat = self.vertical_proj(feat)  # [B, C, W]
        out = self.regressor(feat)    # [B, 1, W]
        out = torch.sigmoid(out.squeeze(1))

        return out
    
class BnetSmallKernelSmarter(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=256
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=decoder_channels,
            activation=None
        )

        self.vertical_proj = HierarchicalVerticalProjection(decoder_channels)

        self.regressor_smarter = nn.Sequential(
            nn.Conv1d(decoder_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        feat = self.unet(x)           # [B, C, H, W]
        feat = self.vertical_proj(feat)  # [B, C, W]
        out=self.regressor_smarter(feat) # [B, 1, W]
        out = torch.sigmoid(out.squeeze(1))

        return out
    
class Refinement1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4), # Normally a 9 was tested 5 and 17
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=9, padding=4), # Same 
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4), # same
        )

    def forward(self, x):
        x = x.unsqueeze(1)   # [B, 1, W]
        x = self.net(x)
        return x.squeeze(1)


class BnetSmallKernelSmarterRefine(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=256
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=decoder_channels,
            activation=None
        )

        self.vertical_proj = HierarchicalVerticalProjection(decoder_channels)

        self.regressor_smarter = nn.Sequential(
            nn.Conv1d(decoder_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1)
        )

        self.refinement=Refinement1D()

    def forward(self, x):
        feat = self.unet(x)           # [B, C, H, W]
        feat = self.vertical_proj(feat)  # [B, C, W]
        coarse = self.regressor_smarter(feat).squeeze(1)   # [B, W]
        delta = self.refinement(coarse)                    # [B, W]
        out = coarse + delta
        out = torch.sigmoid(out)

        return out

    
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1)):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TinyEncoder(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()

        self.net = nn.Sequential(
            DSConv(in_ch, base, stride=(2,1)),     # reduce height only
            DSConv(base, base*2, stride=(2,1)),
            DSConv(base*2, base*2, stride=(2,1)),
        )

        self.out_channels = base * 2

    def forward(self, x):
        return self.net(x)
    
class FastVerticalProjection(nn.Module):
    def forward(self, x):
        # x: [B, C, H, W]
        x = F.adaptive_avg_pool2d(x, (1, None))
        return x.squeeze(2)  # [B, C, W]

class BnetTiny(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = TinyEncoder(in_channels, base=32)
        C = self.encoder.out_channels

        self.vertical_proj = FastVerticalProjection()

        self.regressor = nn.Sequential(
            nn.Conv1d(C, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)          # [B, C, H, W]
        feat = self.vertical_proj(feat) # [B, C, W]
        out = self.regressor(feat)      # [B, 1, W]

        return torch.sigmoid(out.squeeze(1))