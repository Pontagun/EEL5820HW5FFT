import cv2
import numpy as np
from PIL import Image
import math
import time


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        e = fft(x[::2])
        o = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([e + terms[:int(N / 2)] * o,
                               e + terms[int(N / 2):] * o])


def pixel_log255(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            unorm_image[i, j] = (255 / math.log10(256)) * math.log10(1 + (255 / pxmax) * unorm_image[i, j])
            # unorm_image[i, j] = ((unorm_image[i, j] - pxmin) / (pxmax - pxmin)) * 255

    norm_image = unorm_image
    return norm_image


def center_image(image):
    centered_image = np.zeros((rows, cols))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            centered_image[i, j] = image[i, j] * ((-1) ** ((i - 1) + (j - 1)))

    return centered_image


if __name__ == '__main__':
    path = r'img.png'
    img = cv2.imread(path, 0)

    rows = img.shape[0]
    cols = img.shape[1]

    img = center_image(img)
    img_width = rows * cols

    flatten_image = np.zeros(shape=(1, img_width))
    flatten_image = img.flatten()
    fft_image = fft(flatten_image)
    fft_image_2d = np.reshape(fft_image, (rows, cols))
    fft_image_2d = abs(fft_image_2d)
    norm_image = pixel_log255(fft_image_2d)

    im = Image.fromarray(fft_image_2d)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("fft.jpg")
