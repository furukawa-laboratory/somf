import numpy as np
import os



dir_name='beverage_data'
file_name='mixiDrinkData2_Filter123.txt'
#file_path=os.path.join(__file__,file_name)
dir_path=os.path.join(os.path.dirname(__file__),dir_name)#beverage_dataまでのpath
file_path=os.path.join(dir_path,file_name)

file_name1='beverage_label.txt'
file_name2='situation_label2.txt'

label_path1=os.path.join(dir_path,file_name1)
label_path2=os.path.join(dir_path,file_name2)
print(label_path2)
#print(os.path.join(__file__))#こうするとファイルまでのパス
#print(os.path.dirname(__file__))#こうすると実行ファイルがあるディレクトリまでのパス



#実際にloadしてみよう！
data=np.loadtxt(file_path)
x=data.reshape((604,14,11))#reshapeして正し区なっているかは未確認

beverage_label=np.loadtxt(label_path1,dtype='str')
situation_label=np.loadtxt(label_path2,dtype='str',delimiter=',')
print(beverage_label)
print(situation_label)
