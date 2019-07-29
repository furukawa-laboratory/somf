import numpy as np
import os


def load_data(ret_beverage_label=True, ret_situation_label=False):
    dir_name = 'beverage_data'
    file_name = 'beverage_data.npy'
    dir_path = os.path.join(os.path.dirname(__file__), dir_name)  # beverage_dataまでのpath
    file_path = os.path.join(dir_path, file_name)  # path to beverage_data.npy

    x = np.load(file_path)

    return_objects = [x]

    if ret_beverage_label:
        label_name = 'beverage_label.txt'
        label_path = os.path.join(dir_path, label_name)
        beverage_label = np.genfromtxt(label_path, dtype=str)  # loadtxtだと変な文字が入る可能性があるのでgenfromtxt
        return_objects.append(beverage_label)

    if ret_situation_label:
        label_name = 'situation_label.txt'
        label_path = os.path.join(dir_path, label_name)
        situation_label = np.genfromtxt(label_path, dtype=str)  # loadtxtだと変な文字が入る可能性があるのでgenfromtxt
        return_objects.append(situation_label)

    return return_objects
