import torch
import os

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

if __name__ == '__main__':

    diffusion = Diffusion()

    print('\ndiffusion scheduler..')
    print(f'\nnoise steps : {diffusion.noise_steps}')

    print('\nfirst 5 betas:',diffusion.beta[:5])
    print('first 5 alphas:', diffusion.alpha[:5])
    print('first 5 alpha hats:', diffusion.alpha_hat[:5])
    print('last alpha hat:', diffusion.alpha_hat[-1])

    from dataset import create_dataloader
    import matplotlib.pyplot as plt
    
    selected_classes = ['Cat', 'Lion', 'Tiger','Horse', 'Elephant']
    loader = create_dataloader(dataset_path='../animal_data', selected_classes=selected_classes, batch_size=1)
    images, labels = next(iter(loader))

    steps = [0,200,400,600,800,999]

    os.makedirs('../outputs/noisy_samples', exist_ok=True)
    plt.figure(figsize=(15, 3))

    for index, step in enumerate(steps):

        t = torch.tensor([step])
        noisy_image, _ = diffusion.add_noise(images, t)
        image = noisy_image.squeeze(0).permute(1, 2, 0)
        image = (image + 1)/2
        image = image.clamp(0, 1)

        plt.subplot(1, len(steps), index + 1)
        plt.imshow(image)
        plt.title(f't={step}')
        plt.axis('off')

    plt.tight_layout()
    
    plt.savefig('../outputs/noisy_samples/forward_diffusion.png', dpi=300, bbox_inches='tight')
    plt.show()

    from model import DiffusionUNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = Diffusion(device=device)
    model = DiffusionUNet().to(device)
    samples = diffusion.sample(model=model, num_images=2,image_size=64, device=device)

    print('\ntesting reverse diffusion ->', samples.shape, '\n')