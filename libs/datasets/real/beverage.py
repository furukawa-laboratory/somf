import numpy as np
import os


def load_data():
    dir_name='beverage_data'
    file_name='mixiDrinkData2_Filter123.txt'
    dir_path=os.path.join(os.path.dirname(__file__),dir_name)#beverage_dataまでのpath
    #path to mixiDrinkData2_Filter123.txt
    file_path=os.path.join(dir_path,file_name)

    file_name1='beverage_label.txt'
    file_name2='situation_label.txt'

    label_path1=os.path.join(dir_path,file_name1)
    label_path2=os.path.join(dir_path,file_name2)
    print(label_path2)

    temp_data=np.loadtxt(file_path)
    x=temp_data.reshape((604,14,11))#reshapeして正しく(回答者,飲料,状況)に形になっているかは未確認

    beverage_label=np.loadtxt(label_path1,dtype='str',delimiter=',')
    situation_label=np.loadtxt(label_path2,dtype='str',delimiter=',')

    return x,beverage_label,situation_label
