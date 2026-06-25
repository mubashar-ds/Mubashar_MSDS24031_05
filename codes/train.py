import torch
import torch.optim as optim

from dataset import create_dataloader
from diffusion import Diffusion
from model import DiffusionUNet
from loss import DiffusionLoss

import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
batch_size = 8
learning_rate = 1e-3
epochs = 50

def train_one_epoch(model, diffusion, dataloader,optimizer,criterion,device):

    model.train()
    torch.set_grad_enabled(True)

    total_loss = 0.0

    for images, _ in dataloader:

        images = images.to(device)
        timesteps = diffusion.sample_timesteps(images.size(0))
        noisy_images, noise = diffusion.add_noise(images,timesteps)
        predicted_noise = model(noisy_images,timesteps)

        loss = criterion(predicted_noise,noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    return average_loss

def train():

    selected_classes = ['Cat', 'Lion', 'Tiger','Horse', 'Elephant']

    train_loader = create_dataloader(dataset_path='../animal_data', selected_classes=selected_classes, 
                                     batch_size=batch_size, image_size=image_size,images_per_class=20)

    diffusion = Diffusion(device=device)
    model = DiffusionUNet().to(device)
    criterion = DiffusionLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-5)

    loss_history = []
    best_loss = float('inf')

    for epoch in range(epochs):

        average_loss = train_one_epoch(model, diffusion, train_loader,optimizer,criterion,device)
        loss_history.append(average_loss)

        print(f'\nepoch [{epoch + 1}/{epochs}]')
        print(f'\nloss: {average_loss:.3f}')

        if average_loss<best_loss:
            best_loss = average_loss

            os.makedirs('../saved_models', exist_ok=True)
            torch.save(model.state_dict(), '../saved_models/best_diffusion_model.pth')
            print('\nbest model saved,.')

    os.makedirs('../saved_models', exist_ok=True)
    torch.save(model.state_dict(), '../saved_models/diffusion_model.pth')
    print('\nmodel saved..\n')

    plt.figure(figsize=(8,5))
    plt.plot(loss_history, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    os.makedirs('../outputs/plots',exist_ok=True)
    plt.savefig('../outputs/plots/training_loss.png',dpi=300, bbox_inches='tight')
    plt.show()

    return model, loss_history

if __name__ == '__main__':

    model, history = train()