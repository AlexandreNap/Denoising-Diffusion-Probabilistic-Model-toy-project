import torch
from torch import nn


class NoiseModel(nn.Module):
    def __init__(self, n_steps):
        super(NoiseModel, self).__init__()
        self.n_steps = n_steps
        w = 256
        t_w = 16
        self.t_layer = nn.Sequential(nn.Linear(1, 16),
                                  nn.Tanh(),
                                  nn.Linear(16, t_w),
                                  nn.Tanh())
        self.layer1 = nn.Sequential(nn.Linear(t_w + 2, w),
                                    nn.Tanh(),
                                    nn.Linear(w, w),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(w, w),
                                    nn.Tanh(),
                                    nn.Linear(w, w),
                                    nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(w + t_w, w),
                                    nn.Tanh(),
                                    nn.Linear(w, w),
                                    nn.Tanh())

        self.last_layer = nn.Linear(w, 2)

    def forward(self, x, t):
        t = (t.float() / self.n_steps) - 0.5
        temb = self.t_layer(t)

        output = self.layer1(torch.concat([x, temb], axis=-1))
        output = self.layer2(output)
        output = self.layer3(torch.concat([output, temb], axis=-1))
        return self.last_layer(output)
