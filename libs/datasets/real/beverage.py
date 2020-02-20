import numpy as np
import os


def load_data(ret_beverage_label=True, ret_situation_label=False, ret_label_language="english"):
    dir_name = 'beverage_data'
    file_name = 'beverage_data.npy'
    dir_path = os.path.join(os.path.dirname(__file__), dir_name)  # beverage_dataまでのpath
    file_path = os.path.join(dir_path, file_name)  # path to beverage_data.npy

    x = np.load(file_path)

    return_objects = [x]

    if ret_beverage_label:
        if ret_label_language == "english":
            label_name = 'beverage_label.txt'
        elif ret_label_language == "japanese":
            label_name = 'beverage_label_japanese.txt'
        label_path = os.path.join(dir_path, label_name)
        beverage_label = np.genfromtxt(label_path, dtype=str)  # loadtxtだと変な文字が入る可能性があるのでgenfromtxt
        return_objects.append(beverage_label)

    if ret_situation_label:
        if ret_label_language == "english":
            label_name = 'situation_label.txt'
        elif ret_label_language == "japanese":
            label_name = 'situation_label_japanese.txt'
        label_path = os.path.join(dir_path, label_name)
        beverage_label = np.genfromtxt(label_path, dtype=str)  # loadtxtだと変な文字が入る可能性があるのでgenfromtxt
        return_objects.append(beverage_label)

    return return_objects
