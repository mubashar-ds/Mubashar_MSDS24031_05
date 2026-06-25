import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
from PIL import Image

import random
random.seed(42)

import torch.optim as optim

import torch.nn as nn

import matplotlib.pyplot as plt

# ------------------------

class AnimalDataset(Dataset):

    def __init__(self, root_dir, selected_classes, images_per_class=20, image_size=64):
        
        self.samples = []
        self.class_to_idx = {class_name: index for index, class_name in enumerate(selected_classes)}

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

        for class_name in selected_classes:
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            image_names = sorted(os.listdir(class_path))
            random.shuffle(image_names)

            count = 0

            for image_name in image_names:
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                image_path = os.path.join(class_path, image_name)
                self.samples.append((image_path, self.class_to_idx[class_name]))

                count += 1

                if count >= images_per_class:
                    break

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, label

def create_dataloader(dataset_path, selected_classes, batch_size=16, image_size=64, images_per_class=20, shuffle=True):

    dataset = AnimalDataset(root_dir=dataset_path,selected_classes=selected_classes,
                            images_per_class=images_per_class,image_size=image_size)

    loader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)

    return loader

# --------------------------------

class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',):

        self.noise_steps = noise_steps
        self.device = device

        # linear noise schedule..
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)

        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

    def sample_timesteps(self, batch_size):

        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,), device=self.device)
    
    def add_noise(self, images, timesteps):
   
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[timesteps])[:, None,None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[timesteps])[:, None,None,None]

        noise = torch.randn_like(images)
        noisy_images = (sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * noise)

        return noisy_images, noise
    
    @torch.no_grad()
    def sample(self, model, num_images, image_size, device, save_steps=False):

        model.eval()

        x = torch.randn(num_images, 3,image_size,image_size).to(device)

        saved_images = []

        for i in reversed(range(1, self.noise_steps)):

            t = torch.full((num_images,), i, device=device,dtype=torch.long)

            predicted_noise = model(x, t)
            alpha = self.alpha[t][:, None,None, None]
            alpha_hat = self.alpha_hat[t][:,None, None, None]
            beta = self.beta[t][:, None, None,None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

            if save_steps:
                if i in [900,700,500,300,100, 1]:
                    image = (x.clamp(-1, 1) + 1)/2
                    saved_images.append(image.detach().cpu())

        model.train()
        
        x = (x.clamp(-1, 1) + 1) / 2

        if save_steps:
            return x, saved_images

        return x
    
# --------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3,padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8,num_channels=out_channels),
            nn.SiLU(inplace=True))

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
        x = self.bottleneck(x)

        t = t.unsqueeze(-1).unsqueeze(-1)
        x = x + t

        # decoder
        x = self.up1(x,skip2)
        x = self.up2(x,skip1)
        x = self.output_conv(x)

        return x
    
# ---------------------------------

class DiffusionLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self,predicted_noise, target_noise):

        loss = (predicted_noise - target_noise) ** 2

        if self.reduction == 'sum':
            return loss.sum()

        return loss.mean()
    
# --------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
batch_size = 8
learning_rate = 3e-4
epochs = 100

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

# ---------------------------------

