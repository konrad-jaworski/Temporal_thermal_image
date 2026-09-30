import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F 
from torchvision.models import swin_t, Swin_T_Weights
import math

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
    
class BnetSum(nn.Module):
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
        feat = feat.sum(dim=2)  # [B, C, W]

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
            nn.Conv1d(1, 16, kernel_size=5, padding=2), # Normally a 9 was tested 5 and 17
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2), # Same 
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2), # same
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

    
class SwinFeatureEncoder(nn.Module):
    """Pretrained Swin-T features at four spatial resolutions."""

    def __init__(self, pretrained=True):
        super().__init__()

        backbone = swin_t(
            weights=Swin_T_Weights.DEFAULT if pretrained else None
        )

        # Retain the feature extractor, not the classification head.
        self.features = backbone.features
        self.norm = backbone.norm

    def forward(self, x):
        outputs = []

        for index, layer in enumerate(self.features):
            x = layer(x)  # Swin uses (B, H, W, C) internally.

            if index in (1, 3, 5, 7):
                feature = self.norm(x) if index == 7 else x
                outputs.append(feature.permute(0, 3, 1, 2))

        return outputs


class TemporalAttentionPool(nn.Module):
    """Project channels and learn which temporal positions to emphasize."""

    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.score = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        x = self.project(x)                       # (B, D, H, W)
        weights = torch.softmax(self.score(x), dim=2)
        return (x * weights).sum(dim=2)          # (B, D, W)


class CompactTransformerHead(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.pools = nn.ModuleList([
            TemporalAttentionPool(channels, hidden_dim)
            for channels in (96, 192, 384, 768)
        ])

        self.fuse = nn.Sequential(
            nn.Conv1d(4 * hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        # Separate construction gives independently initialized layers.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.regressor = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def positional_encoding(self, length, channels, device, dtype):
        position = torch.arange(
            length, device=device, dtype=torch.float32
        ).unsqueeze(1)

        frequency = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )

        encoding = torch.zeros(length, channels, device=device)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(
            position * frequency[:channels // 2]
        )

        return encoding.unsqueeze(0).to(dtype=dtype)

    def forward(self, features, output_width):
        target_width = features[0].shape[-1]

        pooled = []
        for pool, feature in zip(self.pools, features):
            x = pool(feature)
            x = F.interpolate(
                x,
                size=target_width,
                mode="linear",
                align_corners=False,
            )
            pooled.append(x)

        x = self.fuse(torch.cat(pooled, dim=1))  # (B, D, W/4)
        x = x.transpose(1, 2)                   # (B, W/4, D)

        x = x + self.positional_encoding(
            x.shape[1], x.shape[2], x.device, x.dtype
        )

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x).transpose(1, 2)

        # Restore the original B-scan width before final regression.
        x = F.interpolate(
            x,
            size=output_width,
            mode="linear",
            align_corners=False,
        )

        return torch.sigmoid(self.regressor(x).squeeze(1))


class BnetSwinTransformer(nn.Module):
    """
    Input:  (B, 3, T, W)
    Output: (B, W), normalized depth

    Preserves model.unet.encoder for existing freezing logic.
    """

    def __init__(
        self,
        pretrained=False,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.unet = nn.Module()
        self.unet.encoder = SwinFeatureEncoder(pretrained=pretrained)

        self.head = CompactTransformerHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        return self.head(features, output_width=x.shape[-1])