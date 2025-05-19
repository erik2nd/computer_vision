# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from task1 import show_image, blur_image
from task3 import canny_operator


def hough_transform(image_array):
    height, width = image_array.shape
    diagonal = int(np.sqrt(height ** 2 + width ** 2))
    thetas = np.deg2rad(np.arange(-90, 180, 1))
    rhos = np.linspace(0, diagonal, diagonal)

    phase = np.zeros((len(rhos), len(thetas)), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            if image_array[i, j] > 0:
                for index, theta in enumerate(thetas):
                    rho = int(j * np.cos(theta) + i * np.sin(theta))
                    phase[rho, index] += 1

    return phase, thetas, rhos


def suppress_nonmaximum(phase, threshold):
    max_value = np.max(phase)
    significant_value = int(threshold * max_value)
    phase_height, phase_width = phase.shape

    local_maximum = []
    for r in range(phase_height):
        for t in range(phase_width):
            if phase[r, t] > significant_value:
                neighborhood = phase[max(0, r - 1):min(phase.shape[0], r + 2), max(0, t - 1):min(phase.shape[1], t + 2)]
                if phase[r, t] == np.max(neighborhood):
                    local_maximum.append((r, t, phase[r, t]))

    print(f"Found {len(local_maximum)} lines with threshold {threshold * 100}% from maximum value")

    return local_maximum


def draw_lines(image_array, thetas, rhos, local_maximum, accuracy=0.8):
    height, width, _ = image_array.shape

    for r, t, value in local_maximum:
        theta = thetas[t]
        rho = rhos[r]

        for y in range(height):
            for x in range(width):
                if abs(x * np.cos(theta) + y * np.sin(theta) - rho) < accuracy:
                    image_array[y, x] = [0, 0, 255]

    return image_array


if __name__ == '__main__':
    img = Image.open('images/task.png')
    img_array = np.array(img)
    show_image(img_array)

    canny_img = canny_operator(img_array)
    show_image(canny_img)

    phase_space, thetas_values, rhos_values = hough_transform(canny_img)

    phase_resized = np.repeat(phase_space, 3, axis=1)
    show_image(phase_space)

    blurred_phase_space = blur_image(phase_space, 3, 1)
    maximum = suppress_nonmaximum(blurred_phase_space, 0.6)

    lined_img = draw_lines(img_array, thetas_values, rhos_values, maximum)
    show_image(lined_img)
