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
class DownBlock(nn.Module):

    def __init__(self, in_channels,out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.layers(x)
class UpBlock(nn.Module):

    def __init__(self,in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels,out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([skip,x], dim=1)
        x = self.conv(x)

        return x

if __name__ == '__main__':

    x = torch.randn(2, 3, 64, 64)

    print('\ninput shape:',x.shape)

    double_conv = DoubleConv(3, 64)
    y = double_conv(x)
    print('\nafter DoubleConv:', y.shape)

    down = DownBlock(64, 128)
    y_down= down(y)
    print('after DownBlock:', y_down.shape)

    up = UpBlock(128, 64)
    y_up = up(y_down,y)
    print('after UpBlock;', y_up.shape)

