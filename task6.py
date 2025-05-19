# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from task5 import orb


def hamming_distance(descriptor1, descriptor2):
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(descriptor1, descriptor2))
    return distance


def find_matches_lowe(descriptor1, descriptor2, R_threshold=0.8):
    matches_array = []
    for i, desc1 in enumerate(descriptor1):
        distances = [hamming_distance(desc1, desc2) for desc2 in descriptor2]
        sorted_indices = np.argsort(distances)
        best_match = sorted_indices[0]
        second_best_match = sorted_indices[1]

        R = distances[best_match] / distances[second_best_match]
        if R < R_threshold:
            matches_array.append((i, best_match))
    return matches_array


def cross_check(matches1, matches2):
    cross_checked_matches = []
    for match1 in matches1:
        p1, p2 = match1
        for match2 in matches2:
            if match2 == (p2, p1):
                cross_checked_matches.append(match1)
                break
    return cross_checked_matches


def match_keypoints(descriptors1, descriptors2):
    matches1 = find_matches_lowe(descriptors1, descriptors2)
    matches2 = find_matches_lowe(descriptors2, descriptors1)
    return cross_check(matches1, matches2)


def resize_image_by_percentage(image, percentage):
    original_width, original_height = image.size

    new_width = int(original_width * (percentage / 100))
    new_height = int(original_height * (percentage / 100))

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def calculate_affine_matrix(pts1, pts2):
    A, b = [], []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.extend([[x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1]])
        b.extend([x2, y2])
    A, b = np.array(A), np.array(b)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return params.reshape(2, 3)


def ransac(keypoints1, keypoints2, matches, proj_threshold=3.0):
    best_inliers, best_matrix = [], None

    for i in range(len(matches) - 2):
        for j in range(i + 1, len(matches) - 1):
            for k in range(j + 1, len(matches)):
                sample_matches = [matches[i], matches[j], matches[k]]
                pts1 = np.array([keypoints1[m[0]] for m in sample_matches])
                pts2 = np.array([keypoints2[m[1]] for m in sample_matches])

                try:
                    matrix = calculate_affine_matrix(pts1, pts2)
                except np.linalg.LinAlgError:
                    continue

                transformed_pts1 = np.dot(np.hstack((keypoints1, np.ones((len(keypoints1), 1)))), matrix.T)

                inliers = [m for m in matches if
                           np.linalg.norm(transformed_pts1[m[0]] - keypoints2[m[1]]) < proj_threshold]

                if len(inliers) > len(best_inliers):
                    best_inliers, best_matrix = inliers, matrix

    if best_inliers:
        pts1 = np.array([keypoints1[i] for i, _ in best_inliers])
        pts2 = np.array([keypoints2[j] for _, j in best_inliers])
        best_matrix = calculate_affine_matrix(pts1, pts2)

    return best_matrix, best_inliers


def plot_matches(image1, image2, keypoints1, keypoints2, matches, affine_matrix=None):
    if len(image1.shape) == 2:
        image1 = np.stack([image1] * 3, axis=-1)
    if len(image2.shape) == 2:
        image2 = np.stack([image2] * 3, axis=-1)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = image1
    combined_image[:h2, w1:w1 + w2] = image2

    plt.figure(figsize=(10, 5))
    plt.imshow(combined_image)

    for i, j in matches:
        pt1 = keypoints1[i]
        pt2 = keypoints2[j]
        pt2 = (pt2[0] + w1, pt2[1])
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'y-', lw=0.5)

    if affine_matrix is not None:
        corners = np.array([
            [0, 0, 1], [w1, 0, 1], [w1, h1, 1], [0, h1, 1], [0, 0, 1]
        ])
        transformed_corners = np.dot(corners, affine_matrix.T)
        transformed_corners[:, 0] += w1
        plt.plot(transformed_corners[:, 0], transformed_corners[:, 1], 'r-', lw=2)

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    img_1 = Image.open('images/task.png')
    img_array_1 = np.array(img_1)
    keypoints_1, descriptors_1 = orb(img_1)

    img_2 = Image.open('images/ens.jpg')
    img_array_2 = np.array(img_2)
    keypoints_2, descriptors_2 = orb(img_2)

    points_matches = match_keypoints(descriptors_1, descriptors_2)
    affine_matrix_final, inliers_final = ransac(keypoints_1, keypoints_2, points_matches)
    plot_matches(img_array_1, img_array_2, keypoints_1, keypoints_2, inliers_final, affine_matrix_final)
