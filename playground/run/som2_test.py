import numpy as np
from libs.models.som2 import SOM2
from playground.libs.visualization.som2 import som2_draw_all

#データ読み込み
data = np.load('../../libs/datasets/real/tsom_data/data_all.npy')

model = SOM2(data)
model.fit()

som2_draw_all.draw(data, model.Y, model.pNode_Total_Num, model.cNode_Total_Num)