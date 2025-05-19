# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_image(image_array):
    inverted_image = Image.fromarray(image_array)
    inverted_image.show()


def show_and_save_image(image_array, sigma, size, output_dir='./images'):
    image = Image.fromarray(image_array)
    image.show()
    filename = f"{output_dir}/blurred_image_stddev_{sigma}_size_{size}.png"
    image.save(filename)


def inverse_image(image_array):
    res = np.zeros_like(image_array)
    height, width, channels = image_array.shape
    for i in range(height):
        for j in range(width):
            res[i, j] = 255 - image_array[i, j]
    return res


def grayscale_image(image_array):
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
    else:
        height, width = image_array.shape
    res = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            res[i, j] = np.mean(image_array[i, j]).astype(np.uint8)
    return res


def noise_image(image_array):
    mean = 0
    std_dev = 5
    noise = np.random.normal(mean, std_dev, image_array.shape)
    res = image_array + noise
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res


def build_histogram(image_array):
    global pixel_counts
    height, width = image_array.shape
    pixel_counts.fill(0)
    for i in range(height):
        for j in range(width):
            value = image_array[i, j]
            pixel_counts[value] += 1

    plt.bar(range(256), pixel_counts)
    plt.title('Absolute frequencies of image pixels')
    plt.show()


def build_cumulative_histogram():
    global cumulative_counts
    cumulative_counts.fill(0)
    cumulative_sum = 0
    for i in range(len(pixel_counts)):
        cumulative_sum += pixel_counts[i]
        cumulative_counts[i] = cumulative_sum

    plt.bar(range(256), cumulative_counts)
    plt.title('Absolute cumulative frequencies of image pixels')
    plt.show()


def G(x, y, size, sigma):
    center = size // 2
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))


def gaussian_kernel(size, sigma):
    res = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            res[i][j] = G(i, j, size, sigma)
    return res / np.sum(res)


def blur_image(image_array, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    pad = 0 if size - 2 <= 0 else size - 2
    padded_image = np.pad(image_array, pad_width=pad, mode='edge')
    center = size // 2
    res = np.zeros_like(image_array)
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
    else:
        height, width = image_array.shape

    for i in range(height):
        for j in range(width):
            mul_sum = 0
            for k in range(size):
                for p in range(size):
                    x = (i + pad) + k - center
                    y = (j + pad) + p - center
                    mul_sum += padded_image[x, y] * kernel[k, p]

            res[i, j] = mul_sum

    return res.astype(np.uint8)


def equalize_image(image_array, cdf):
    height, width = image_array.shape
    pixels = height * width
    res = np.zeros((height, width), dtype=np.uint8)
    colors = np.zeros_like(cdf)
    length = len(cdf)
    cdf_min = cdf[cdf > 0].min()
    for i in range(length):
        colors[i] = round((cdf[i] - cdf_min) / (pixels - 1) * 255)

    for i in range(height):
        for j in range(width):
            res[i, j] = colors[image_array[i, j]]

    return res


if __name__ == '__main__':
    img = Image.open('images/task.png')
    img_array = np.array(img)

    inverted_img_array = inverse_image(img_array)
    show_image(inverted_img_array)

    grayscale_img_array = grayscale_image(img_array)
    show_image(grayscale_img_array)

    # noised_image = noise_image(grayscale_img_array)
    noised_image = grayscale_img_array
    show_image(noised_image)

    pixel_counts = np.zeros(256, dtype=int)
    cumulative_counts = np.zeros_like(pixel_counts)
    build_histogram(noised_image)
    build_cumulative_histogram()

    size_value = 3
    sigma_value = 5
    blurred_image = blur_image(noised_image, size_value, sigma_value)
    # show_and_save_image(blurred_image)
    show_image(blurred_image)

    build_histogram(blurred_image)
    build_cumulative_histogram()

    equalized_image = equalize_image(blurred_image, cumulative_counts)
    show_image(equalized_image)
    build_histogram(equalized_image)

    # img = Image.open('images/test.png')
    # img_array = np.array(img)
    # height, width, channels = img_array.shape
    # pixels = height * width
    #
    # grayscale_img_array = grayscale_image(img_array)
    # show_image(grayscale_img_array)
    #
    #
    # pixel_counts = np.zeros(256, dtype=int)
    # cumulative_counts = np.zeros_like(pixel_counts)
    #
    # size = 9
    # sigma = 5
    #
    # build_histogram(grayscale_img_array)
    # build_cumulative_histogram()
    #
    # equalized_image = equalize_image(grayscale_img_array, cumulative_counts)
    # show_image(equalized_image)
    # build_histogram(equalized_image)
