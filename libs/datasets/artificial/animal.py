import numpy as np
import os


def load_data(retlabel=True):
    datastore_name = 'datastore/animal'
    file_name = 'features.txt'

    directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    file_path = os.path.join(directory_path, file_name)

    x = np.loadtxt(file_path)

    if retlabel:
        label_name = 'labels_animal.txt'
        label_path = os.path.join(directory_path, label_name)
        label = np.genfromtxt(label_path, dtype=str)

        return x, label
    else:
        return x
