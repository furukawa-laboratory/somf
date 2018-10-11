import numpy as np
import os




dir_name='beverage_data'
file_name='mixiDrinkData2_Filter123.txt'
#file_path=os.path.join(__file__,file_name)
dir_path=os.path.join(os.path.dirname(__file__),dir_name)#beverage_dataまでのpath
file_path=os.path.join(dir_path,file_name)


#print(os.path.join(__file__))#こうするとファイルまでのパス
#print(os.path.dirname(__file__))#こうすると実行ファイルがあるディレクトリまでのパス



#実際にloadしてみよう！
data=np.loadtxt(file_path)
x=data.reshape((604,14,11))
print(x.shape)
