# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque


def show_image(image_array):
    inverted_image = Image.fromarray(image_array)
    inverted_image.show()


def show_and_save_image(image_array, output_dir='./images/task2'):
    image = Image.fromarray(image_array)
    image.show()
    filename = f"{output_dir}/segmented_image14.png"
    image.save(filename)


def grayscale_image(image_array):
    res = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            res[i, j] = np.mean(image_array[i, j]).astype(np.uint8)
    return res


def build_histogram(image_array):
    global pixel_counts, probabilities
    pixel_counts.fill(0)
    for i in range(height):
        for j in range(width):
            value = image_array[i, j]
            pixel_counts[value] += 1

    probabilities = pixel_counts / pixels


def class_probabilities(t):
    return (np.sum(probabilities[:t + 1]),
            np.sum(probabilities[t + 1:]))


def means(q1, q2, t):
    return (np.sum(np.arange(t + 1) * probabilities[:t + 1]) / q1,
            np.sum(np.arange(t + 1, 256) * probabilities[t + 1:]) / q2)


def variances(mu1, mu2, q1, q2, t):
    return (np.sum(((np.arange(t + 1) - mu1) ** 2) * probabilities[:t + 1]) / q1,
            np.sum(((np.arange(t + 1, 256) - mu2) ** 2) * probabilities[t + 1:]) / q2)


def optimal_threshold_inside(image_array):
    build_histogram(image_array)
    t = 1
    u = t
    T = u
    s_min = np.inf

    while u < 256:
        weight1, weight2 = class_probabilities(u)
        if weight1 == 0 or weight2 == 0:
            u += t
            continue

        mean1, mean2 = means(weight1, weight2, u)
        variance1, variance2 = variances(mean1, mean2, weight1, weight2, u)
        weighted_variance = weight1 * variance1 + weight2 * variance2

        if weighted_variance < s_min:
            s_min = weighted_variance
            T = u

        u += t

    return T


def optimal_threshold_between(image_array):
    build_histogram(image_array)
    t = 1
    u = t
    T = u
    s_max = 0

    while u < 256:
        weight1, weight2 = class_probabilities(u)
        if weight1 == 0 or weight2 == 0:
            u += t
            continue

        mean1, mean2 = means(weight1, weight2, u)
        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2

        if variance_between > s_max:
            s_max = variance_between
            T = u

        u += t

    return T


def binarize_image(image_array):
    threshold = optimal_threshold_between(image_array)
    print(threshold)
    res = np.where(image_array >= threshold, 255, 0).astype(np.uint8)
    return res


def remove_salt_and_pepper(image_array):
    image_copy = image_array.copy()
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = image_copy[i - 1:i + 2, j - 1:j + 2]
            center = window[1, 1]
            window[1, 1] = not center
            if image_copy[i, j] == 0 and np.all(window == 255):
                image_array[i, j] = 1
            elif image_copy[i, j] == 255 and np.all(window == 0):
                image_array[i, j] = 0


def is_valid(x, y):
    return 0 <= x < height and 0 <= y < width


def segment_binary_image(image_array):
    res = np.zeros_like(image_array, dtype=int)
    segment_id = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(height):
        for j in range(width):
            if res[i, j] == 0:
                queue = deque([(i, j)])
                res[i, j] = segment_id

                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if is_valid(nx, ny) and res[nx, ny] == 0:
                            if image_array[x, y] == image_array[nx, ny]:
                                res[nx, ny] = segment_id
                                queue.append((nx, ny))

                segment_id += 1

    return res


def segment_grayscale_image(image_array, cluster_ranges):
    res = np.zeros_like(image_array, dtype=int)
    segment_id = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for cluster_range in cluster_ranges:
        for i in range(height):
            for j in range(width):
                if res[i, j] == 0 and cluster_range[0] <= image_array[i, j] <= cluster_range[1]:
                    queue = deque([(i, j)])
                    res[i, j] = segment_id

                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in directions:
                            nx = x + dx
                            ny = y + dy
                            if is_valid(nx, ny) and res[nx, ny] == 0:
                                if cluster_range[0] <= image_array[nx, ny] <= cluster_range[1]:
                                    res[nx, ny] = segment_id
                                    queue.append((nx, ny))
                    segment_id += 1

    return res


def create_colored_image(labels):
    res = np.zeros((height, width, 3), dtype=np.uint8)
    segment_ids = np.unique(labels)
    segment_colors = {}
    for segment_id in segment_ids:
        segment_colors[segment_id] = [random.randint(0, 255) for _ in range(3)]
    for i in range(height):
        for j in range(width):
            segment_id = labels[i, j]
            res[i, j] = segment_colors[segment_id]
    return res


def create_gradient_colored_image(labels, start_color=(0, 0, 0), end_color=(255, 255, 255)):
    res = np.zeros((height, width, 3), dtype=np.uint8)
    segment_ids = np.unique(labels)
    num_segments = len(segment_ids)
    print(num_segments)
    segment_colors = {}
    for idx, segment_id in enumerate(segment_ids):
        t = idx / (num_segments - 1) if num_segments > 1 else 1
        r = int((1 - t) * start_color[0] + t * end_color[0])
        g = int((1 - t) * start_color[1] + t * end_color[1])
        b = int((1 - t) * start_color[2] + t * end_color[2])
        segment_colors[segment_id] = (r, g, b)

    for i in range(height):
        for j in range(width):
            segment_id = labels[i, j]
            res[i, j] = segment_colors[segment_id]

    return res


def histogram_clustering(image_array, size=10):
    build_histogram(image_array)
    total = len(pixel_counts)
    min_indices = []

    for start in range(0, total, size):
        end = min(start + size, total)
        local_min = None
        local_min_value = np.inf
        for i in range(start + 1, end - 1):
            if pixel_counts[i] < pixel_counts[i - 1] and pixel_counts[i] < pixel_counts[i + 1]:
                if pixel_counts[i] < local_min_value:
                    local_min_value = pixel_counts[i]
                    local_min = i

        if local_min is not None:
            min_indices.append(local_min)

    cluster_ranges = []
    for i in range(1, len(min_indices)):
        cluster_ranges.append((min_indices[i - 1], min_indices[i]))
    if min_indices:
        cluster_ranges.insert(0, (0, min_indices[0]))
        cluster_ranges.append((min_indices[-1], 255))

    return cluster_ranges


def plot_histogram_with_clusters(cluster_ranges):
    plt.bar(range(256), pixel_counts, width=1, color='gray', alpha=0.6, label='Гистограмма')
    for start, end in cluster_ranges:
        plt.axvline(x=start, color='red')
        plt.axvline(x=end, color='red')

    plt.title('Гистограмма с границами кластеров')
    plt.show()


if __name__ == '__main__':
    img = Image.open('images/task.png')
    img_array = np.array(img)
    height, width, channels = img_array.shape
    pixels = height * width
    pixel_counts = np.zeros(256, dtype=int)
    probabilities = np.zeros(256, dtype=int)

    grayscale_img_array = grayscale_image(img_array)

    binary_image = binarize_image(grayscale_img_array)
    show_image(binary_image)

    remove_salt_and_pepper(binary_image)
    show_image(binary_image)

    segments = segment_binary_image(binary_image)
    colored_image = create_colored_image(segments)
    # colored_image = create_gradient_colored_image(segments, (0, 255, 0))
    show_image(colored_image)

    clusters = histogram_clustering(grayscale_img_array, 100)
    plot_histogram_with_clusters(clusters)
    segments = segment_grayscale_image(grayscale_img_array, clusters)
    colored_image = create_colored_image(segments)
    # colored_image = create_gradient_colored_image(segments, (0, 0, 255))
    show_image(colored_image)
