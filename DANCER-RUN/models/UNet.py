# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class EncBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 padding=(kernel_size - 1) // 2,
#             ),
#             nn.LeakyReLU(),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class DecBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, act=True):
#         super().__init__()
#         self.act = act
#         self.conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 padding=(kernel_size - 1) // 2,
#             ),
#         )
#         if act:
#             self.relu = nn.LeakyReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         if self.act:
#             x = self.relu(x)

#         return x


# class UNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         channels = [2, 16, 32, 64, 128]
#         kernel_size = [13, 7, 7, 7]
#         self.encoder = nn.ModuleList()
#         self.decoder = nn.ModuleList()

#         self.bottle_neck = nn.Sequential(
#             nn.Conv1d(
#                 channels[-1],
#                 channels[-1],
#                 kernel_size=3,
#                 padding=1,
#             ),
#             nn.LeakyReLU(),
#         )
#         self.down = nn.MaxPool1d(2)
#         self.up = nn.Upsample(scale_factor=2, mode="linear")

#         for i in range(4):
#             self.encoder.append(
#                 EncBlock(
#                     in_channels=channels[i],
#                     out_channels=channels[i + 1],
#                     kernel_size=kernel_size[i],
#                 )
#             )
#             self.decoder.append(
#                 DecBlock(
#                     in_channels=channels[-(i + 1)],
#                     out_channels=channels[-(i + 2)],
#                     kernel_size=kernel_size[-(i + 1)],
#                     act=True if i != 3 else False,
#                 )
#             )

#     def forward(self, x):
#         encfeature = []
#         for i in range(4):
#             x = self.encoder[i](x)
#             encfeature.append(x)
#             x = self.down(x)

#         x = self.bottle_neck(x)

#         for i in range(4):
#             x = self.up(x)
#             x += encfeature[-(i + 1)]
#             x = self.decoder[i](x)
#         return x


# if __name__ == "__main__":
#     x = torch.rand(16, 2, 256)
#     model = UNet()
#     print(model)
#     y = model(x)
#     print(y.shape)



# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=True):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        ]
        if act:
            layers.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        """
        U-Net for single-channel EEG denoising.

        Args:
            in_channels (int): Number of input channels (1 for EEG).
            base_channels (int): Number of channels in the first encoder layer.
        """
        super().__init__()

        # Channel progression: [1, 16, 32, 64, 128]
        channels = [in_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        kernel_size = [13, 7, 7, 7]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        # Bottleneck
        self.bottle_neck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        # Build encoder and decoder blocks
        for i in range(4):
            self.encoder.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size[i],
                )
            )
            self.decoder.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],      # from upsample or bottleneck
                    out_channels=channels[-(i + 2)],     # target channel after skip connection
                    kernel_size=kernel_size[-(i + 1)],
                    act=(i != 3),                        # no activation on final output
                )
            )

    def forward(self, x):
        enc_features = []
        # Encoder with skip connections
        for i in range(4):
            x = self.encoder[i](x)
            enc_features.append(x)
            x = self.down(x)

        # Bottleneck
        x = self.bottle_neck(x)

        # Decoder with skip connections
        for i in range(4):
            x = self.up(x)
            x = x + enc_features[-(i + 1)]  # element-wise addition
            x = self.decoder[i](x)

        return x


if __name__ == "__main__":
    # Test with single-channel EEG-like input: (batch, channel=1, time=256)
    x = torch.randn(16, 1, 256)
    model = UNet(in_channels=1)
    print(model)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")  # Should be (16, 1, 256)
    
