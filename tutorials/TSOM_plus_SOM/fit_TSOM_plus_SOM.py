# 人工データをplus型階層TSOM(TSOM*SOM)に適用したプログラムを追加する.
import numpy as np
from libs.datasets.artificial.kura_tsom import load_kura_tsom
import matplotlib.pyplot as plt
from libs.models.TSOM_plus_SOM import TSOM_plus_SOM
from mpl_toolkits.mplot3d import Axes3D
from libs.visualization.som.Grad_norm import Grad_Norm

# 人工データの検証
xsamples = 40  # x_samplesでメンバーの人数が100人になるように調整
ysamples = 20
X, z = load_kura_tsom(xsamples=xsamples, ysamples=ysamples, retz=True)

# チームを分割(偶数チームのみ)
group_num = 8
input_data = np.zeros((group_num, int(xsamples * ysamples / group_num), 3))  # グループ数*メンバー数*次元
# 観測データを分割
for i in range(int(group_num / 2)):
    group1 = X[int(xsamples * i / int(group_num / 2)):int(xsamples * (i + 1) / int(group_num / 2)), 0:int(ysamples / 2),
             :]
    group1 = group1.reshape((int(xsamples * ysamples / group_num), 3))
    group2 = X[int(xsamples * i / int(group_num / 2)):int(xsamples * (i + 1) / int(group_num / 2)),
             int(ysamples / 2):int(ysamples), :]
    group2 = group2.reshape((int(xsamples * ysamples / group_num), 3))

    input_data[int(2 * i), :, :] = group1
    input_data[int(2 * i + 1), :, :] = group2

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection="3d")
# for i in range(group_num):
#     ax.scatter(input_data[i, :, 0], input_data[i, :, 1], input_data[i, :, 2])
# plt.show()
input_data = input_data.reshape(-1,3)

# グループラベルの作成
group_label = []
for i in range(group_num):
    group_label_i = np.arange(int(100 * (i)), int(100 * (i + 1)))
    group_label.append(group_label_i)

params_tsom = {'latent_dim':[2,2],
               'resolution':[10,10],
               'SIGMA_MAX':[1.0,1.0],
               'SIGMA_MIN':[0.1,0.1],
               'TAU':[50,50]}
params_som = {'latent_dim':2,
              'resolution':10,
              'sigma_max':2.0,
              'sigma_min':0.5,
              'tau':50,
              'init':'random'}

# +型階層TSOMのclass読み込み
# group_label以降の変数ははlatent_dim,resolution,sigma_max,sigma_min,tauでSOMとTSOMでまとめている
tsom_plus_som = TSOM_plus_SOM(input_data=input_data,
                              group_label=group_label,
                              params_tsom=params_tsom,
                              params_som=params_som)

tsom_plus_som.fit_1st_TSOM(tsom_epoch_num=50)
tsom_plus_som.fit_KDE(kernel_width=1.0)
tsom_plus_som.fit_2nd_SOM(som_epoch_num=250)  # 2ndSOMの学習

# grad_normでteamを可視化
som_umatrix = Grad_Norm(X=tsom_plus_som.som.X, Z=tsom_plus_som.som.Z, sigma=0.1, labels=None, title_text="team_map",
                        resolution=10)
som_umatrix.draw_umatrix()
