import numpy as np
import tensorflow as tf
import cv2
from dh_segment.inference import LoadedModel

def convert_to_desired_cmap_basic(array):
    colorized_array = np.zeros_like(array)

    colors = [
        [0, 38, 255],
        [255, 48, 0],
        [170, 121, 66],
        [255, 255, 255]]

    orig_colors = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]]

    colorized_array[np.all(array == orig_colors[0], axis=-1)] = colors[0]
    colorized_array[np.all(array == orig_colors[1], axis=-1)] = colors[1]
    colorized_array[np.all(array == orig_colors[2], axis=-1)] = colors[2]
    colorized_array[np.all(array == orig_colors[3], axis=-1)] = colors[3]

    return colorized_array

def convert_to_desired_cmap_advanced(array):
    colorized_array = np.zeros_like(array)

    colors = [
        [0, 38, 255],
        [255, 48, 0],
        [170, 121, 66],
        [233, 255, 0],
        [255, 255, 255]]

    orig_colors = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]]

    colorized_array[np.all(array == orig_colors[0], axis=-1)] = colors[0]
    colorized_array[np.all(array == orig_colors[1], axis=-1)] = colors[1]
    colorized_array[np.all(array == orig_colors[2], axis=-1)] = colors[2]
    colorized_array[np.all(array == orig_colors[3], axis=-1)] = colors[3]
    colorized_array[np.all(array == orig_colors[4], axis=-1)] = colors[4]

    return colorized_array


def find_arg_max(probs):
    probs_shape = probs.shape
    decision = np.argmax(probs, axis=-1)

    return decision

def save_array_to_image(array, filename):
    cv2.imwrite(filename, array)

def make_prediction(pred, task_sel):

    if task_sel == "basic":
        return convert_to_desired_cmap_basic(cv2.merge([find_arg_max(pred), find_arg_max(pred), find_arg_max(pred)]))    
    elif task_sel == "advanced":
        return convert_to_desired_cmap_advanced(cv2.merge([find_arg_max(pred), find_arg_max(pred), find_arg_max(pred)]))    
    
def predict(model_dir, input_image_filename, task_sel):
    with tf.Session():

        model = LoadedModel(model_dir, predict_mode='filename_original_shape')
        prediction_outputs = model.predict(input_image_filename)
        pred = prediction_outputs['probs'][0]
        prediction = make_prediction(pred, task_sel)

        return prediction

def blend_images(original, mask):

    alpha = 0.65
    beta = 1.0 - alpha
    gamma = 0.0

    original = np.asarray(original, dtype = np.int32)
    mask = np.asarray(mask, dtype = np.int32)

    # print(original.dtype)
    # print(mask.dtype)

    blended = cv2.addWeighted(mask, alpha, original, beta, gamma)

    return blended