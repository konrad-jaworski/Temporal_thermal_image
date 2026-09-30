"""
B-net architectures: networks that predict a depth profile from a thermal
B-scan.

Input:  B-scan x of shape [B, 3, 512, 512]
        (batch, 3 identical channels, time, position along the scan line).
Output: depth profile of shape [B, 512], one value per position, in (0, 1)
        (normalised depth; 0 = no defect).

All variants share the same idea:
  1. A U-Net (segmentation_models_pytorch, ImageNet-pretrained encoder)
     turns the B-scan into a feature map [B, C, H, W] of the same size.
     Here H is the time axis and W the spatial axis.
  2. The time axis is collapsed, leaving one feature vector per position
     [B, C, W]. The variants differ in how this is done.
  3. A 1D head maps every feature vector to one depth value, followed by a
     sigmoid.

Variants:
  BnetMean                     - time axis averaged; per-position head.
  BnetSmallKernel              - time axis compressed by a learned stack of
                                 vertical convolutions
                                 (HierarchicalVerticalProjection).
  BnetSmallKernelSmarter       - as above, with a head that also looks at
                                 neighbouring positions.
  BnetSmallKernelSmarterRefine - as above, plus a residual 1D refinement of
                                 the predicted profile.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BnetMean(nn.Module):
    """
    Simplest B-net: U-Net features averaged over time, then a per-position
    regression.

    Parameters
    ----------
    encoder_name : str
        Encoder of the U-Net, any name supported by
        segmentation_models_pytorch (default ResNet-34).
    encoder_weights : str or None
        Pretrained weights of the encoder ("imagenet" or None).
    in_channels : int
        Number of input channels (3, the B-scan is repeated to match the
        pretrained encoder).
    decoder_channels : int
        Number of feature channels C produced by the U-Net for every pixel.
    output_width : int
        Width of the predicted profile. Stored only; the output width is
        set by the input width.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=256,
        output_width=512
    ):
        super().__init__()

        # The U-Net is used as a feature extractor, not as a segmentation
        # network: its output "classes" are decoder_channels feature maps at
        # full input resolution, without an activation.
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=decoder_channels,
            activation=None
        )

        # Kernel size 1 in a Conv1d means the same small MLP
        # (C -> 128 -> 1) is applied to every position independently.
        self.regressor = nn.Sequential(
            nn.Conv1d(decoder_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, 3, 512, 512]
        return: [B, 512]
        """

        feat = self.unet(x)          # [B, C, H, W]

        # Average over the time axis: each position keeps the mean of its
        # features over the whole sequence.
        feat = feat.mean(dim=2)      # [B, C, W]

        out = self.regressor(feat)   # [B, 1, W]

        out = out.squeeze(1)         # [B, W]

        # Sigmoid keeps the prediction in (0, 1), the range of the
        # normalised depth targets.
        out = torch.sigmoid(out)

        return out



class HierarchicalVerticalProjection(nn.Module):
    """
    Learned compression of the time axis of a feature map to length 1.

    Instead of simply averaging over time, four convolutions with a tall,
    one-pixel-wide kernel (15 x 1) and stride 2 along time halve the time
    axis step by step:
        512 -> 256 -> 128 -> 64 -> 32
    A last convolution with a 32 x 1 kernel then combines the remaining 32
    time steps into one value, with learned weights. Each position (column)
    is processed on its own (kernel width 1), so no information is mixed
    between positions here; that is already done by the U-Net.

    Because of the fixed 32 x 1 kernel at the end, the input height must be
    512 (512 / 2^4 = 32).
    """
    def __init__(self, channels):
        super().__init__()

        # padding=(7, 0) with a 15-tall kernel and stride 2 gives exactly
        # half the height at each step.
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

            # Height 32 -> 1.
            nn.Conv2d(channels, channels, kernel_size=(32, 1))
        )

    def forward(self, x):
        x = self.net(x)       # [B, C, H, W] -> [B, C, 1, W]
        return x.squeeze(2)   # [B, C, W]

class BnetSmallKernel(nn.Module):
    """
    B-net with a learned time compression (HierarchicalVerticalProjection)
    in place of the plain mean used in BnetMean. The head is the same
    per-position MLP.

    Parameters are the same as for BnetMean (without output_width).
    """
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
        feat = self.unet(x)              # [B, C, H, W]
        feat = self.vertical_proj(feat)  # [B, C, W]
        out = self.regressor(feat)       # [B, 1, W]
        out = torch.sigmoid(out.squeeze(1))   # [B, W]

        return out

class BnetSmallKernelSmarter(nn.Module):
    """
    BnetSmallKernel with a larger regression head.

    The first two layers of the head use kernel size 3, so the depth at each
    position also depends on its neighbours (receptive field of 5
    positions). This helps to give a smooth, consistent depth across a
    defect. Batch normalisation after the first layer stabilises training.

    Parameters are the same as for BnetSmallKernel.
    """
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

        # padding=1 with kernel 3 keeps the width unchanged.
        self.regressor_smarter = nn.Sequential(
            nn.Conv1d(decoder_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        feat = self.unet(x)                # [B, C, H, W]
        feat = self.vertical_proj(feat)    # [B, C, W]
        out=self.regressor_smarter(feat)   # [B, 1, W]
        out = torch.sigmoid(out.squeeze(1))   # [B, W]

        return out

class Refinement1D(nn.Module):
    """
    Small 1D CNN that works on a predicted depth profile [B, W] and returns
    a correction of the same shape.

    Three convolutions with kernel size 5 give each output a view of 13
    neighbouring positions, enough to correct local errors such as noisy
    values inside a defect or blurred defect edges.
    """
    def __init__(self):
        super().__init__()
        # Kernel size 5 is used; 9 and 17 were also tested.
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)   # [B, W] -> [B, 1, W]
        x = self.net(x)      # [B, 1, W]
        return x.squeeze(1)  # [B, W]


class BnetSmallKernelSmarterRefine(nn.Module):
    """
    BnetSmallKernelSmarter followed by a residual refinement step.

    The head first gives a coarse profile. Refinement1D predicts a
    correction, which is added to it (residual connection), and the sigmoid
    is applied only to the sum. The refinement therefore only has to learn
    the difference from the coarse prediction, not the whole profile.

    Parameters are the same as for BnetSmallKernel.
    """
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
        feat = self.unet(x)                                # [B, C, H, W]
        feat = self.vertical_proj(feat)                    # [B, C, W]
        # The coarse profile is kept before the sigmoid (logits), so the
        # correction is added in the unbounded space.
        coarse = self.regressor_smarter(feat).squeeze(1)   # [B, W]
        delta = self.refinement(coarse)                    # [B, W]
        out = coarse + delta
        out = torch.sigmoid(out)

        return out