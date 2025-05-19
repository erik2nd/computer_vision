# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw

from task1 import show_image, grayscale_image, blur_image
from task3 import gradient, sobel_kernel


def draw_keypoints(image, keypoints):
    if image.mode != 'RGB':
        img_copy = image.convert('RGB')
    else:
        img_copy = image.copy()

    draw = ImageDraw.Draw(img_copy)

    for x, y in keypoints:
        radius = 5
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline="red", width=2)

    img_copy.show()


def bresenham_circle_offsets():
    return [(0, -3), (1, -3), (2, -2), (3, -1),
            (3, 0), (3, 1), (2, 2), (1, 3),
            (0, 3), (-1, 3), (-2, 2), (-3, 1),
            (-3, 0), (-3, -1), (-2, -2), (-1, -3)]


def bresenham_circle(image_array, x, y):
    offsets = bresenham_circle_offsets()
    return [image_array[x + dx, y + dy] for dx, dy in offsets]


def check_point(image_array, x, y, I_p, threshold):
    circle_values = bresenham_circle(image_array, x, y)

    # Добавим полтора круга для анализа
    extended_circle_values = circle_values + circle_values[:len(circle_values) // 2]

    greater, smaller = 0, 0

    for i in range(len(extended_circle_values)):
        if extended_circle_values[i] > I_p + threshold:
            greater += 1
            smaller = 0
        elif extended_circle_values[i] < I_p - threshold:
            smaller += 1
            greater = 0
        else:
            greater = 0
            smaller = 0

        if greater >= 12 or smaller >= 12:
            return True
    return False


def check_opposite_points(image_array, x, y, I_p, threshold):
    circle_offsets = bresenham_circle_offsets()

    point_1 = image_array[x + circle_offsets[0][0], y + circle_offsets[0][1]]
    point_9 = image_array[x + circle_offsets[8][0], y + circle_offsets[8][1]]

    if ((point_1 < I_p - threshold and point_9 < I_p - threshold) or
            (point_1 > I_p + threshold and point_9 > I_p + threshold)):
        point_5 = image_array[x + circle_offsets[4][0], y + circle_offsets[4][1]]
        point_13 = image_array[x + circle_offsets[12][0], y + circle_offsets[12][1]]

        if ((point_5 < I_p - threshold and point_13 < I_p - threshold) or
                (point_5 > I_p + threshold and point_13 > I_p + threshold)):
            return True

    return False


def fast(image_array, threshold):
    height, width = image_array.shape
    keypoints = []

    for i in range(3, height - 3):
        for j in range(3, width - 3):
            I_p = int(image_array[i, j])
            if check_opposite_points(image_array, i, j, I_p, threshold):
                if check_point(image_array, i, j, I_p, threshold):
                    keypoints.append((j, i))

    return keypoints


def build_matrices_for_keypoints(image_array, keypoints, kernel_size=5, sigma=1.0):
    I_x, I_y = gradient(image_array, sobel_kernel())
    matrices = []

    for p in keypoints:
        x, y = p

        window_size = kernel_size
        x_min, x_max = max(x - window_size, 0), min(x + window_size, image_array.shape[1] - 1)
        y_min, y_max = max(y - window_size, 0), min(y + window_size, image_array.shape[0] - 1)

        I_x_2 = I_x[y_min:y_max + 1, x_min:x_max + 1] ** 2
        I_y_2 = I_y[y_min:y_max + 1, x_min:x_max + 1] ** 2
        I_x_y = I_x[y_min:y_max + 1, x_min:x_max + 1] * I_y[y_min:y_max + 1, x_min:x_max + 1]

        S_x_x = blur_image(I_x_2, kernel_size, sigma)[window_size, window_size]
        S_y_y = blur_image(I_y_2, kernel_size, sigma)[window_size, window_size]
        S_x_y = blur_image(I_x_y, kernel_size, sigma)[window_size, window_size]

        M = np.array([[S_x_x, S_x_y], [S_x_y, S_y_y]])
        matrices.append(M)

    return matrices


def harris_response(M, k=0.04):
    det_M = np.linalg.det(M)
    trace_M = np.trace(M)
    R = det_M - k * (trace_M ** 2)
    return R


def filter_by_harris_responses(image_array, keypoints, method='top_n', N=100, T=0.7, k=0.04, M_grid_size=50, M_limit=3):
    matrices = build_matrices_for_keypoints(image_array, keypoints)
    R_values = []

    for M, p in zip(matrices, keypoints):
        R = harris_response(M, k)
        R_values.append((R, p))

    R_values.sort(key=lambda x: x[0], reverse=True)
    selected_points = []

    if method == 'top_n':
        selected_points = [kp for _, kp in R_values[:N]]

    elif method == 'threshold':
        selected_points = [kp for R, kp in R_values if R > T]

    elif method == 'grid':
        grid_points = {}
        for R, p in R_values:
            grid_x = p[0] // M_grid_size
            grid_y = p[1] // M_grid_size
            if (grid_x, grid_y) not in grid_points:
                grid_points[(grid_x, grid_y)] = []
            grid_points[(grid_x, grid_y)].append(p)

        selected_points = []
        for points in grid_points.values():
            selected_points.extend(points[:M_limit])

    return selected_points


def calculate_moment(image_array, x_center, y_center, p, q, radius):
    moment = 0
    for x in range(x_center - radius, x_center + radius + 1):
        for y in range(y_center - radius, y_center + radius + 1):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius**2:
                if 0 <= x < image_array.shape[1] and 0 <= y < image_array.shape[0]:
                    moment += (x ** p) * (y ** q) * int(image_array[y, x])
    return moment


def calculate_orientations(image, keypoints, radius=31):
    orientations = []

    for (x, y) in keypoints:
        m10 = calculate_moment(image, x, y, p=1, q=0, radius=radius)
        m01 = calculate_moment(image, x, y, p=0, q=1, radius=radius)

        if m10 != 0 or m01 != 0:
            angle = np.atan2(m01, m10)
            orientations.append(angle)
        else:
            orientations.append(0)

    return orientations


def generate_pairs(size=15, count=256):
    mean = 0
    std_dev = size ** 2 / 25

    matrix = np.zeros((2, count, 2), dtype=int)

    for k in range(count):
        x1 = int(np.clip(np.abs(np.random.normal(mean, std_dev)), 0, size - 1))
        y1 = int(np.clip(np.abs(np.random.normal(mean, std_dev)), 0, size - 1))

        x2 = int(np.clip(np.abs(np.random.normal(mean, std_dev)), 0, size - 1))
        y2 = int(np.clip(np.abs(np.random.normal(mean, std_dev)), 0, size - 1))

        matrix[0, k] = [x1, y1]
        matrix[1, k] = [x2, y2]

    final_matrix = matrix.reshape(2, count * 2)
    np.save("pairs.npy", final_matrix)
    return final_matrix


def load_pairs_from_file(filename="pairs.npy"):
    return np.load(filename)


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_points(points, theta, size):
    rot_matrix = rotation_matrix(theta)
    rotated_points = np.dot(rot_matrix, points)
    rotated_points = np.round(rotated_points).astype(int)

    rotated_points[0] = np.clip(rotated_points[0], 0, size - 1)
    rotated_points[1] = np.clip(rotated_points[1], 0, size - 1)

    return rotated_points


def brief_descriptor(image, keypoints, orientations, patch_size=15, count=256):
    descriptors = []
    pairs = load_pairs_from_file()
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    padded_image = np.pad(image, 15, mode='edge')

    for (x, y), angle in zip(keypoints, orientations):
        y_min = y - patch_size // 2 + patch_size
        y_max = y + patch_size // 2 + 1 + patch_size
        x_min = x - patch_size // 2 + patch_size
        x_max = x + patch_size // 2 + 1 + patch_size

        patch = padded_image[y_min:y_max, x_min:x_max]

        rotated_point_sets = []
        for theta in angles:
            rotated_point_sets.append(rotate_points(pairs, theta, patch_size))

        closest_angle_idx = np.argmin(np.abs(angles - angle))
        rotated_pairs = rotated_point_sets[closest_angle_idx]

        descriptor = np.zeros(count, dtype=int)

        for i in range(count):
            x1, y1 = rotated_pairs[0, i], rotated_pairs[1, i]
            x2, y2 = rotated_pairs[0, i + count], rotated_pairs[1, i + count]

            if patch[y1, x1] < patch[y2, x2]:
                descriptor[i] = 1
            else:
                descriptor[i] = 0

        descriptors.append(descriptor)

    return descriptors


def orb(image):
    image_array = np.array(image)

    gray_img = grayscale_image(image_array)
    # show_image(gray_img)

    keypoints = fast(gray_img, 30)
    draw_keypoints(image, keypoints)

    # filter_points = filter_by_harris_responses(gray_img, keypoints, 'threshold')
    # draw_keypoints(image, filter_points)

    orientations = calculate_orientations(gray_img, keypoints)
    blurred_img = blur_image(gray_img, 5, 3)
    descriptors = brief_descriptor(blurred_img, keypoints, orientations)

    return keypoints, descriptors


if __name__ == "__main__":
    img = Image.open('images/task.png')
    img_array = np.array(img)

    grayscale_img = grayscale_image(img_array)
    show_image(grayscale_img)

    t = 20
    key_points = fast(grayscale_img, t)
    print(f'Keypoints count: {len(key_points)}')
    draw_keypoints(img, key_points)

    filtered_points = filter_by_harris_responses(grayscale_img, key_points, 'grid')
    print(f'Filtered keypoints count: {len(filtered_points)}')
    draw_keypoints(img, filtered_points)

    orients = calculate_orientations(grayscale_img, filtered_points)
    blurred_image = blur_image(grayscale_img, 5, 3)

    descriptor_vectors = brief_descriptor(blurred_image, filtered_points, orients)
    for index, d in enumerate(descriptor_vectors):
        print(f"Descriptor {index + 1}: {d}")

    with open("descriptor_vectors.txt", "w") as file:
        for d in descriptor_vectors:
            file.write(" ".join(map(str, d)) + "\n")

    print("Дескрипторы успешно сохранены в файл 'descriptor_vectors.txt'")
