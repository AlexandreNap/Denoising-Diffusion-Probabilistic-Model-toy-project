import gradio as gr
from src.toy_dataset import array_to_toy_dataset
from src.training import train
import torch
from matplotlib import pyplot as plt
import numpy as np

batch_size = 1024
n_epochs = 100
n_steps = 100
lr = 0.001
device = "cuda"


def train_and_denoise(img):
    print(img.shape)
    img = 255 - np.asarray(img)
    array = img / img.sum()
    data = array_to_toy_dataset(array, 10000)
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    diffuser, losses = train(data, batch_size, n_epochs, n_steps, lr, device)

    eval_data = torch.randn(512, 2)
    diffuser.model.eval()
    with torch.no_grad():
        _, all_outputs = diffuser.denoise(torch.Tensor(eval_data).to(device), n_steps)

    fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(16, 12),
                            constrained_layout=True)
    axs = axs.flatten()
    k = 0
    n_plots = 20

    for i, output in enumerate(all_outputs):
        if i % 5 == 0:
            if k < n_plots:
                axs[k].scatter(output.cpu().numpy()[:, 0], output.cpu().numpy()[:, 1])
                k += 1

    return fig


gr.Interface(fn=train_and_denoise,
             inputs=gr.components.Image(image_mode="L", source="canvas", shape=None),
             outputs="plot",
             live=False).launch()
