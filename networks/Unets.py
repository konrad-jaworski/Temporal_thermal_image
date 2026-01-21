import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


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

#-----------------------------------------------------------------------------------------------------------------------------
class VerticalProjection(nn.Module):
    """
    Learnable projection from [B, C, H, W] -> [B, C, W]
    using a convolution across the H dimension.
    """
    def __init__(self, channels, height):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(height, 1),
            bias=False
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)      # [B, C, 1, W]
        return x.squeeze(2)  # [B, C, W]


class BnetBigKernel(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=256,
        input_height=512,
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

        # --- Learnable vertical projection ---
        self.vertical_proj = VerticalProjection(
            channels=decoder_channels,
            height=input_height
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
        returns: [B, 512] with values in [0,1]
        """

        # U-Net output: [B, C, H, W]
        feat = self.unet(x)

        # Learnable vertical compression: [B, C, W]
        feat = self.vertical_proj(feat)

        # Column-wise regression: [B, 1, W]
        out = self.regressor(feat)

        # [B, W]
        out = out.squeeze(1)

        # Enforce depth range [0,1]
        out = torch.sigmoid(out)

        return out

#-----------------------------------------------------------------------------------------------------------------------------

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