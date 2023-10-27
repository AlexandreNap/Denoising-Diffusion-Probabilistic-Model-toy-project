import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.diffusion import DiffusionModel
from src.model import NoiseModel
from src.ema import EMA


def train(data,  beta_1, beta_t, batch_size, n_epochs, n_steps, lr, device):
    my_dataset = TensorDataset(torch.Tensor(data))
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    model = NoiseModel(n_steps)
    diffuser = DiffusionModel(model, n_steps, beta_1, beta_t, device)

    ema = EMA(0.9)
    ema.register(diffuser.model)

    optimizer = torch.optim.Adam(diffuser.model.parameters(), lr=lr, weight_decay=0.001)

    diffuser.model.train()
    all_losses = []

    for epoch in tqdm(range(n_epochs)):
        for batch in my_dataloader:
            batch = batch[0].to(device)
            t = torch.randint(1, diffuser.n_steps + 1, (len(batch),)).unsqueeze(1).to(device)
            eps, diffused = diffuser.diffuse(batch, t)
            pred_eps = model(diffused, t)

            loss = (eps - pred_eps) ** 2
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            loss = loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema.update(model)
            all_losses.append(loss)

    ema.ema(model)
    return diffuser, all_losses
