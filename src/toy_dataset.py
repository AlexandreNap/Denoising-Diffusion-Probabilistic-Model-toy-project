from PIL import Image, ImageFilter
import numpy as np


def array_to_toy_dataset(array, n_samples):
    """
    Create dataset from ditribution 2d array
    """
    # Create a flat copy of the array
    flat = array.flatten()

    # Then, sample an index from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat, size=n_samples)

    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, array.shape)
    data = np.array([[x, y] for x, y in zip(adjusted_index[1], array.shape[1] - adjusted_index[0])])
    return data


def image_to_toy_dataset(path, n_samples, blur_radius=1):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = img.convert('P')
    img = 255 - np.asarray(img)
    array = img / img.sum()
    return array_to_toy_dataset(array, n_samples)
