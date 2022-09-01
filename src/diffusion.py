import torch


class DiffusionModel:
    def __init__(self, model, n_steps, beta_1, beta_t, device):
        """
        Denoising Diffusion Probabilistic Model diffusion model
        Personnal light implementation according to https://arxiv.org/pdf/2006.11239.pdf
        :param torch.nn.Module: model used to predict noise of diffused images
        :param int n_steps: total number of diffusion steps
        :param float beta_1: initial beta for beta scheduler
        :param float beta_t: last beta for beta scheduler
        :param string device: torch device to place tensors and model on
        """
        self.model = model.to(device)
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_1, beta_t, self.n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)

        self.r_alphas_bar = torch.sqrt(self.alphas_bar)
        self.r_1m_alphas_bar = torch.sqrt(1 - self.alphas_bar)

        self.inv_r_alphas = torch.pow(self.alphas, -0.5)
        self.pre_noise_terms = self.betas / self.r_1m_alphas_bar
        self.sigmas = torch.pow(self.betas, 0.5)

        self.device = device

    def diffuse(self, x, t):
        """
        Diffuse x for t steps.
        :param torch.Tensor x: 2d data points to diffuse
        :param torch.Tensor t: number of diffusion time steps
        :return: diffused data points
        """
        eps = torch.randn(x.shape).to(self.device)
        t = t - 1
        diffused = self.r_alphas_bar[t] * x + self.r_1m_alphas_bar[t] * eps
        return eps, diffused

    def denoise(self, x, t):
        """
        Denoise random samples x for t steps.
        :param torch.Tensor x: initial 2d data points to denoise
        :param torch.Tensor t: number of denoising time steps
        :return torch.tensor, list: (denoised data points, list of each denoised data points for all diffusion steps)
        """
        n_samples = 1
        if len(x.shape)>1:
            n_samples = x.shape[0]
        all_x = [x]
        for i in range(t, 0, -1):
            z = torch.randn(x.shape).to(self.device)
            if i == 1:
                z = z * 0
            steps = torch.full((n_samples,), i, dtype=torch.int, device=self.device).unsqueeze(1)
            model_output = self.model(x, steps)
            x = self.inv_r_alphas[i - 1] * (x - self.pre_noise_terms[i - 1] * model_output) + self.sigmas[i - 1] * z
            all_x.append(x)
        return x, all_x
