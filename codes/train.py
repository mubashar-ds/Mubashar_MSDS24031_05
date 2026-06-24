import torch
import torch.optim as optim

from dataset import create_dataloader
from diffusion import Diffusion
from model import DiffusionUNet
from loss import DiffusionLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
batch_size = 8
learning_rate = 1e-3

selected_classes = ['Cat', 'Lion', 'Tiger','Horse', 'Elephant']


train_loader = create_dataloader(dataset_path='../animal_data', selected_classes=selected_classes, 
                                 batch_size=batch_size, image_size=image_size, images_per_class=20)

diffusion = Diffusion(device=device)

model = DiffusionUNet().to(device)
criterion = DiffusionLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

images, _ = next(iter(train_loader))
images = images.to(device)
timesteps = diffusion.sample_timesteps(images.shape[0])
noisy_images, noise = diffusion.add_noise(images, timesteps)

predicted_noise = model(noisy_images, timesteps)
loss = criterion(predicted_noise, noise)

optimizer.zero_grad()
loss.backward()
optimizer.step()

if __name__ == '__main__':

    print('\ntesting training pipeline,..')

    print(f'\nimagess shape : {images.shape}')
    print(f'noisy shape: {noisy_images.shape}')
    print(f'prediction shape : {predicted_noise.shape}')

    print(f'loss : {loss.item():.5f}')
    print('\nbackward pass successful..\n')