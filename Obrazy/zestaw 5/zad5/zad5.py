import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import generic_filter
from PIL import Image
import matplotlib.pyplot as plt

image_path = "./meduza.png"
image = Image.open(image_path).convert("L")
image_array = np.array(image)


def mid_range_filter(values):
    return (np.max(values) + np.min(values)) / 2

def trimmed_mean_filter(values, k):
    sorted_values = np.sort(values)
    trimmed_values = sorted_values[k:-k]
    return np.mean(trimmed_values)

def k_nearest_neighbor_filter(values, k):

    sorted_values = np.sort(values)
    return np.mean(sorted_values[:k])

median_filtered = median_filter(image_array, size=3)
mid_range_filtered = generic_filter(image_array, mid_range_filter, size=3)
trimmed_mean_filtered = generic_filter(image_array, trimmed_mean_filter, size=3, extra_arguments=(2,))
k_nearest_filtered = generic_filter(image_array, k_nearest_neighbor_filter, size=3, extra_arguments=(6,))

def symmetric_nearest_neighbour(values):
    sorted_values = np.sort(values)
    symmetric_mean = np.mean(sorted_values[4:6])  
    return symmetric_mean


symmetric_nn_filtered = generic_filter(image_array, symmetric_nearest_neighbour, size=3)


def normalize_to_uint8(img):
    min_val = np.min(img)
    max_val = np.max(img)
    norm_img = (img - min_val) / (max_val - min_val) * 255
    return norm_img.astype(np.uint8)

median_filtered_norm = normalize_to_uint8(median_filtered)
mid_range_filtered_norm = normalize_to_uint8(mid_range_filtered)
trimmed_mean_filtered_norm = normalize_to_uint8(trimmed_mean_filtered)
k_nearest_filtered_norm = normalize_to_uint8(k_nearest_filtered)
symmetric_nn_filtered_norm = normalize_to_uint8(symmetric_nn_filtered)

#zapisz
Image.fromarray(median_filtered_norm).save("meduza_median_norm.png")
Image.fromarray(mid_range_filtered_norm).save("meduza_mid_range_norm.png")
Image.fromarray(trimmed_mean_filtered_norm).save("meduza_trimmed_mean_norm.png")
Image.fromarray(k_nearest_filtered_norm).save("meduza_k_nearest_norm.png")
Image.fromarray(symmetric_nn_filtered_norm).save("meduza_symmetric_nn_norm.png")


