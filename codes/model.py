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
    
class TimeEmbedding(nn.Module):

    def __init__(self,embedding_dim):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self,t):

        t = t.float().unsqueeze(1)

        return self.embedding(t)
    
class DiffusionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # time embedding...
        self.time_embedding = TimeEmbedding(256)

        # encoder..
        self.input_conv = DoubleConv(3,64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)

        # bottleneck..
        self.bottleneck = DoubleConv(256,256)

        # decoder..
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128,64)

        # final output..
        self.output_conv = nn.Conv2d(64, 3,kernel_size=1)

    def forward(self, x, t):

        # time embedding..
        t = self.time_embedding(t)

        # encoder
        skip1 = self.input_conv(x)
        skip2 = self.down1(skip1)
        x = self.down2(skip2)

        # bottleneck
        x =self.bottleneck(x)

        # injecting timestep information..
        t = t.unsqueeze(-1).unsqueeze(-1)

        x = x + t

        # decoder
        x = self.up1(x,skip2)
        x = self.up2(x, skip1)
        x = self.output_conv(x)

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

    time_embedding = TimeEmbedding(256)

    t = torch.tensor([10, 500])
    emb = time_embedding(t)
    print('time Embedding:', emb.shape)

    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([100, 700])

    model = DiffusionUNet()

    output = model(x, t)

    print('\ntesting diffusion UNet..')

    print(f'input Shape : {x.shape}')
    print(f'output Shape : {output.shape}\n')
