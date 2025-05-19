# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from collections import deque
from task1 import show_image, grayscale_image, blur_image


def prewitt_kernel():
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    prewitt_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])
    return prewitt_x, prewitt_y


def sobel_kernel():
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    return sobel_x, sobel_y


def gradient(image_array, kernel):
    kernel_x, kernel_y = kernel
    padded_image = np.pad(image_array, pad_width=1, mode='reflect')
    grad_x = np.zeros_like(image_array, dtype=np.float64)
    grad_y = np.zeros_like(image_array, dtype=np.float64)

    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
    else:
        height, width = image_array.shape
    kernel_size = kernel_x.shape[0]
    center = kernel_size // 2

    for i in range(height):
        for j in range(width):
            x_sum, y_sum = 0, 0
            for k in range(kernel_size):
                for p in range(kernel_size):
                    x = (i + 1) + k - center
                    y = (j + 1) + p - center
                    x_sum += padded_image[x, y] * kernel_x[k, p]
                    y_sum += padded_image[x, y] * kernel_y[k, p]

            grad_x[i, j] = x_sum
            grad_y[i, j] = y_sum

    # grad_x_img = Image.fromarray(np.uint8(np.clip(grad_x, 0, 255)))
    # grad_y_img = Image.fromarray(np.uint8(np.clip(grad_y, 0, 255)))
    # grad_x_img.show()
    # grad_y_img.show()

    return grad_x, grad_y


def magnitude(grad_x, grad_y):
    # grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = np.abs(grad_x) + np.abs(grad_y)

    # result_image = Image.fromarray(grad_magnitude.astype(np.uint8))
    # result_image.show()

    return grad_magnitude


def direction(grad_x, grad_y):
    grad_direction = np.arctan2(grad_y, grad_x)
    grad_direction_deg = (np.degrees(grad_direction) + 360) % 360

    # grad_direction_img = (grad_direction_deg / 360 * 255).astype(np.uint8)
    # grad_direction_img_pil = Image.fromarray(grad_direction_img)
    # grad_direction_img_pil.show()

    offset = 22.5
    grad_direction_round = (np.floor((grad_direction_deg + offset) / 45) * 45) % 360
    return grad_direction_round


def suppress_nonmaximum(grad_magnitude, grad_direction):
    height, width = grad_magnitude.shape
    result = np.zeros((height, width), dtype=np.float32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = grad_direction[i, j] % 180

            first, second = 0, 0
            if angle == 0 or angle == 180:
                first = grad_magnitude[i, j + 1]
                second = grad_magnitude[i, j - 1]
            elif angle == 45:
                first = grad_magnitude[i - 1, j + 1]
                second = grad_magnitude[i + 1, j - 1]
            elif angle == 90:
                first = grad_magnitude[i - 1, j]
                second = grad_magnitude[i + 1, j]
            elif angle == 135:
                first = grad_magnitude[i + 1, j + 1]
                second = grad_magnitude[i - 1, j - 1]

            if grad_magnitude[i, j] >= first and grad_magnitude[i, j] >= second:
                result[i, j] = grad_magnitude[i, j]

    return result


def hysteresis(grad_magnitude, t_low, t_high):
    height, width = grad_magnitude.shape
    result = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    result[grad_magnitude > t_high] = 255
    result[grad_magnitude < t_low] = 0

    for i in range(height):
        for j in range(width):
            if grad_magnitude[i, j] > t_high and not visited[i, j]:
                visited[i, j] = True
                result[i, j] = 255
                queue = deque([(i, j)])
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                            if grad_magnitude[nx, ny] > t_low:
                                visited[nx, ny] = True
                                result[nx, ny] = 255
                                queue.append((nx, ny))

    return result


def canny_operator(image_array):
    grayscale_img = grayscale_image(image_array)
    blurred_img = blur_image(grayscale_img, 3, 1)

    grad_x, grad_y = gradient(blurred_img, sobel_kernel())
    grad_magnitude = magnitude(grad_x, grad_y)
    grad_direction = direction(grad_x, grad_y)

    suppress_img = suppress_nonmaximum(grad_magnitude, grad_direction)
    hysteresis_img = hysteresis(suppress_img, 100, 200)

    return hysteresis_img


if __name__ == '__main__':
    img = Image.open('images/task.png')
    img_array = np.array(img)

    grayscale_image = grayscale_image(img_array)
    show_image(grayscale_image)

    blurred_image = blur_image(grayscale_image, 3, 1)
    show_image(blurred_image)

    gradient_x, gradient_y = gradient(blurred_image, sobel_kernel())
    gradient_magnitude = magnitude(gradient_x, gradient_y)
    gradient_direction = direction(gradient_x, gradient_y)

    suppress = suppress_nonmaximum(gradient_magnitude, gradient_direction)
    show_image(suppress)

    hysteresis_image = hysteresis(suppress, 100, 200)
    image = Image.fromarray(hysteresis_image)
    image.show()
    filename = f"images/task3/canny_image_chess.png"
    image.save(filename)
