import torch
import torch.nn as nn

class DiffusionLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self,predicted_noise, target_noise):

        loss = (predicted_noise - target_noise) ** 2

        if self.reduction == 'sum':
            return loss.sum()

        return loss.mean()
    
if __name__ == '__main__':

    print('\ntesting fiffusion loss..')

    batch_size = 4
    channels = 3
    height = 64
    width = 64

    predicted_noise = torch.randn(batch_size, channels, height, width)
    target_noise = torch.randn(batch_size, channels, height, width)

    criterion = DiffusionLoss()
    loss = criterion(predicted_noise, target_noise)

    print(f'\nloss value: {loss.item():.5f} \n')