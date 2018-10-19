import numpy as np
import os


def load_data(ret_beverage_label=True,ret_situation_label=False):
    dir_name='beverage_data'
    file_name='mixiDrinkData2_Filter123.txt'
    dir_path=os.path.join(os.path.dirname(__file__),dir_name)#beverage_dataまでのpath
    #path to mixiDrinkData2_Filter123.txt
    file_path=os.path.join(dir_path,file_name)

    temp_data = np.loadtxt(file_path)
    x = temp_data.reshape((604, 14, 11))  # reshapeして正しく(回答者,飲料,状況)に形になっているかは未確認

    return_objects=[x]

    if ret_beverage_label:
        label_name='beverage_label.txt'
        label_path = os.path.join(dir_path, label_name)
        beverage_label = np.genfromtxt(label_path, dtype=str)  # loadtxtだと変な文字が入る可能性があるのでgenfromtxt
        return_objects.append(beverage_label)

    if ret_situation_label:
        label_name='situation_label.txt'
        label_path = os.path.join(dir_path, label_name)
        situation_label = np.genfromtxt(label_path, dtype=str)  # loadtxtだと変な文字が入る可能性があるのでgenfromtxt
        return_objects.append(situation_label)

    return return_objects
