import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        return self.layers(x)

if __name__ == '__main__':

    x = torch.randn(2, 3, 64, 64)

    print('\ninput shape')
    print(x.shape)

    double_conv = DoubleConv(3, 64)
    y = double_conv(x)

    print('\nafter DoubleConv')
    print(y.shape)

