# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from itertools import combinations


def print_values_example():
    random_indices = np.random.choice(len(digits.images), size=5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, idx in enumerate(random_indices):
        axes[i].imshow(digits.images[idx], cmap=plt.cm.gray)
        axes[i].set_title(f"Label: {digits.target[idx]}")
        axes[i].axis('off')
    plt.show()


def inside_cluster_distance_sum(data, labels, centers, clusters_count=10):
    distance = 0

    for i in range(clusters_count):
        cluster_points = data[labels == i]
        cluster_center = centers[i]
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        squared_distances = distances ** 2
        distance += np.sum(squared_distances)

    return distance


def between_cluster_distance_sum(centers, clusters_count=10):
    distance = 0

    for i in range(clusters_count):
        for j in range(i + 1, clusters_count):
            inter_distance = np.linalg.norm(centers[i] - centers[j])
            squared_inter_distance = inter_distance ** 2
            distance += squared_inter_distance

    return distance


def inside_cluster_distance_mean(data, labels, clusters_count=10):
    distance = 0
    pairs = 0

    for i in range(clusters_count):
        cluster_points = data[labels == i]
        num_points = len(cluster_points)

        if num_points > 1:
            for point1, point2 in combinations(cluster_points, 2):
                distance = np.linalg.norm(point1 - point2)
                distance += distance
                pairs += 1

    mean_distance = distance / pairs if pairs > 0 else 0
    return mean_distance


def between_cluster_distance_mean(data, labels, clusters_count=10):
    distance = 0
    pairs = 0

    for i in range(clusters_count):
        for j in range(i + 1, clusters_count):
            cluster_i_points = data[labels == i]
            cluster_j_points = data[labels == j]

            for point1 in cluster_i_points:
                for point2 in cluster_j_points:
                    dist = np.linalg.norm(point1 - point2)
                    distance += dist
                    pairs += 1

    mean_distance = distance / pairs if pairs > 0 else 0
    return mean_distance


def print_distance_metrics(data, labels, centers):
    inside_sum = inside_cluster_distance_sum(data, labels, centers)
    print(f"Сумма средних внутрикластерных расстояний: {inside_sum}")
    # print(kmeans.inertia_)
    between_sum = between_cluster_distance_sum(centers)
    print(f"Сумма межкластерных растояний: {between_sum}")
    print(f"Отношение функционалов: {inside_sum / between_sum}")

    inside_mean = inside_cluster_distance_mean(data, labels)
    print(f"Среднее внутрикластерное расстояние: {inside_mean}")
    between_mean = between_cluster_distance_mean(data, labels)
    print(f"Среднее межкластерное расстояние: {between_mean}")
    print(f"Отношение функционалов: {inside_mean / between_mean}")


def mark_clusters(labels, target, clusters_count=10):
    cluster_to_digit = {}
    for cluster in range(clusters_count):
        cluster_points_indices = np.where(labels == cluster)[0]
        cluster_true_labels = target[cluster_points_indices]
        most_common_label = np.bincount(cluster_true_labels).argmax()
        cluster_to_digit[cluster] = most_common_label

    found_labels = np.array([cluster_to_digit[label] for label in labels])
    return found_labels


def print_class_metrics(target_labels, found_labels, clusters_count=10):
    conf_matrix = confusion_matrix(target_labels, found_labels)
    print("\nМатрица ошибок:")
    print(conf_matrix)

    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    precision = np.zeros(clusters_count)  
    for i in range(clusters_count):
        d = np.sum(conf_matrix[:, i])
        if d != 0:
            precision[i] = conf_matrix[i, i] / d

    recall = np.zeros(clusters_count)
    for i in range(clusters_count):
        d = np.sum(conf_matrix[i, :])
        if d != 0:
            recall[i] = conf_matrix[i, i] / d

    print("\nМетрики качества:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision по классам: {precision}")
    print(f"Recall по классам: {recall}")


def print_detailed_info(target_labels, found_labels):
    print("-" * 60)
    print("№ изображения | Истинная метка | Предсказанная метка | Угадано")
    print("-" * 60)

    correct_count = 0
    for i in range(len(target_labels)):
        correct = "Да" if target_labels[i] == found_labels[i] else "Нет"
        print(f"{i:<13} | {target_labels[i]:<14} | {found_labels[i]:<18} | {correct}")
        if correct == "Да":
            correct_count += 1

    print('\n')
    print(f"Верно: {correct_count}")
    print(f"Неверно: {digits.images.shape[0] - correct_count}")


def raw_pixels(data):
    return data


def intensity_histogram(data, bins=16):
    histograms = np.array([np.histogram(image, bins=bins, range=(0, 16))[0] for image in data])
    return histograms


def gradient_histogram(data, bins=16):
    histograms = []
    for image in data:
        image_2d = image.reshape(8, 8)
        gradients = np.gradient(image_2d)
        gradient_magnitudes = np.sqrt(gradients[0]**2 + gradients[1]**2)
        histogram, _ = np.histogram(gradient_magnitudes, bins=bins, range=(0, np.max(gradient_magnitudes)))
        histograms.append(histogram)
    return np.array(histograms)


def direction_histogram(data, bins=16):
    histograms = []
    for image in data:
        image_2d = image.reshape(8, 8)
        gradients = np.gradient(image_2d)
        gradient_directions = np.arctan2(gradients[1], gradients[0])
        histogram, _ = np.histogram(gradient_directions, bins=bins, range=(-np.pi, np.pi))
        histograms.append(histogram)
    return np.array(histograms)


def block_average_intensity(data, block_size=(4, 4)):
    block_features = []
    for image in data:
        image_2d = image.reshape(8, 8)
        features = []
        for i in range(0, image_2d.shape[0], block_size[0]):
            for j in range(0, image_2d.shape[1], block_size[1]):
                block = image_2d[i:i+block_size[0], j:j+block_size[1]]
                features.append(np.mean(block))
        block_features.append(features)
    return np.array(block_features)


def intensity_statistics(data):
    statistics = []
    for image in data:
        mean_intensity = np.mean(image)
        median_intensity = np.median(image)
        max_intensity = np.max(image)
        statistics.append([mean_intensity, median_intensity, max_intensity])
    return np.array(statistics)


def block_intensity_histograms(data, block_size=(4, 4), bins=8):
    all_histograms = []
    for image in data:
        image_2d = image.reshape(8, 8)
        histograms = []
        for i in range(0, image_2d.shape[0], block_size[0]):
            for j in range(0, image_2d.shape[1], block_size[1]):
                block = image_2d[i:i+block_size[0], j:j+block_size[1]]
                histogram, _ = np.histogram(block, bins=bins, range=(0, 16))
                histograms.extend(histogram)
        all_histograms.append(histograms)
    return np.array(all_histograms)


def split_into_fragments(data, num_rows=2, num_cols=2):
    fragment_features = []
    for image in data:
        image_2d = image.reshape(8, 8)
        row_step = image_2d.shape[0] // num_rows
        col_step = image_2d.shape[1] // num_cols
        fragments = []
        for i in range(0, image_2d.shape[0], row_step):
            for j in range(0, image_2d.shape[1], col_step):
                fragment = image_2d[i:i+row_step, j:j+col_step]
                fragments.append(np.mean(fragment))
        fragment_features.append(fragments)
    return np.array(fragment_features)


def split_into_9_fragments(data):
    return split_into_fragments(data, num_rows=3, num_cols=3)


def split_into_25_fragments(data):
    return split_into_fragments(data, num_rows=5, num_cols=5)


def run_kmeans(characteristic):
    values = characteristic(digits.data)
    target = digits.target

    kmeans = KMeans(n_clusters=10, random_state=50)
    kmeans.fit(values)

    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    print_distance_metrics(values, cluster_labels, cluster_centers)

    predicted_labels = mark_clusters(cluster_labels, target)
    print_class_metrics(target, predicted_labels)

    # print_detailed_info(target, predicted_labels)


if __name__ == '__main__':
    digits = load_digits()
    # print_values_example()

    print("Изображения:")
    run_kmeans(raw_pixels)

    print("\nГистограммы интенсивностей:")
    run_kmeans(intensity_histogram)

    print("\nГистограммы градиента:")
    run_kmeans(gradient_histogram)

    print("\nГистограммы направлений градиента:")
    run_kmeans(direction_histogram)

    print("\nСредние интенсивности по фрагментам:")
    run_kmeans(block_average_intensity)

    print("\nСтатистики интенсивностей (cреднее, медиана, максимум):")
    run_kmeans(intensity_statistics)

    print("\nГистограммы значений пикселей по фрагментам:")
    run_kmeans(block_intensity_histograms)

    print("\nИзображения по фрагментам (4):")
    run_kmeans(split_into_fragments)

    print("\nИзображения по фрагментам (9):")
    run_kmeans(split_into_9_fragments)

    print("\nИзображения по фрагментам (25):")
    run_kmeans(split_into_25_fragments)
