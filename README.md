# Denoising Diffusion Probabilistic Model toy project

This repository contains an implementation of Diffusion model from 
"Denoising Diffusion Probabilistic Models" paper : https://arxiv.org/abs/2006.11239*

Instead of applying diffusion model on images, we apply it on differents 2d data
points distibutions.
Distributions are created from black and white images stored in `images/`
by `image_to_toy_dataset()` from `src` directory.
