import torch
class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',):

        self.noise_steps = noise_steps
        self.device = device

        # linear noise schedule..
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)

        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

if __name__ == '__main__':

    diffusion = Diffusion()

    print('\ndiffusion scheduler..')

    print(f'\nnoise steps : {diffusion.noise_steps}')

    print('\nfirst 5 betas:',diffusion.beta[:5])
    print('first 5 alphas:', diffusion.alpha[:5])
    print('first 5 alpha hats:', diffusion.alpha_hat[:5])
    print('last alpha hat:', diffusion.alpha_hat[-1], '\n')
