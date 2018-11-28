import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d as axes3d


def update(t, x, y, ax, epoch_num, xz, yz, ax_num):
    if t >= epoch_num:
        t = epoch_num - 1
    # キャンパスのクリア
    for i in range(ax_num):
        ax.cla()

    # タイトルの設定
    ax.set_title('t=' + str(t))

    # データ表示
    for i in range(x.shape[0]):
        ax.scatter(x[i][:, 0], x[i][:, 1], xz[:], marker='x', c='blue', s=2)

    # 参照ベクトル表示
    magnification = 1.0
    for i in range(ax_num):
        ax.plot_wireframe(y[i][t, :, :, 0], y[i][t, :, :, 1], yz[:, :], linewidths=1, color='r')
        # ax[i].scatter(y[i][t, :, 0], y[i][t, :, 1], yz[:], marker='o', c='red', s=4)
        ax.view_init(elev=90, azim=270)
        # ax[i].set_xlim(y[i][t, :, 0:2].min() * magnification, y[i][t, :, 0:2].max() * magnification)
        # ax[i].set_ylim(y[i][t, :, 0:2].min() * magnification, y[i][t, :, 0:2].max() * magnification)
        # ax[i].set_xlim(y[i][t, :, :, 0:2].min() * magnification, y[i][t, :, :, 0:2].max() * magnification)
        # ax[i].set_ylim(y[i][t, :, :, 0:2].min() * magnification, y[i][t, :, :, 0:2].max() * magnification)
        ax.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        ax.tick_params(labelright="off", right="off")  # y軸の削除
        ax.axis('off')


    # 描画範囲とカメラアングル
    # ax.set_xlim(x[:, 0:2].min() * magnification, x[:, 0:2].max() * magnification)
    # ax.set_ylim(x[:, 0:2].min() * magnification, x[:, 0:2].max() * magnification)
    # ax[i].view_init(elev=90, azim=270)


def draw(x, y, parent_node_num, child_node_num):
    # ユニットの一辺を計算
    fig_num = int(np.sqrt(parent_node_num))
    resolution = int(np.sqrt(child_node_num))

    # キャンバスの作成
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # メッシュを作成
    y_mesh = []
    for i in range(parent_node_num):
        #
        temp_y = np.copy(y[:, i, :])
        temp2_y = temp_y.reshape(temp_y.shape[0], child_node_num, x.shape[2])
        y_mesh.append(temp2_y.reshape((temp_y.shape[0], resolution, resolution, x.shape[2])))
        # y_mesh.append(temp2_y)


    # Z軸を作成
    data_z_axis = np.zeros(x.shape[1])
    y_z_axis = np.zeros([y_mesh[0].shape[1], y_mesh[0].shape[2]])
    # y_z_axis = np.zeros(y_mesh[0].shape[1])


    # 描画
    ani = animation.FuncAnimation(fig, update, fargs=(x, y_mesh, ax, y.shape[0], data_z_axis, y_z_axis, parent_node_num), interval=10, frames=y.shape[0]*5, repeat=False)
    plt.show()


if __name__ == '__main__':
    # データ読み込み
    X = np.loadtxt("./data/data1.txt")
    # Y = np.load("y.npy")
    #
    # # 描画
    # Fig = plt.figure(figsize=(8, 4))
    # Ax1 = Fig.add_subplot(1, 1, 1, projection='3d')
    # Y_mesh = Y.reshape((Y.shape[0], 10, 10, Y.shape[2]))
    #
    # Data_z_axis = np.zeros(X.shape[0])
    # Y_z_axis = np.zeros([Y_mesh.shape[1], Y_mesh.shape[2]])
    # # y_z_axis = np.zeros(Y.shape[1])
    #
    # Ani = animation.FuncAnimation(Fig, update, fargs=(X, Y_mesh, Ax1, Y.shape[0], Data_z_axis, Y_z_axis), interval=10, frames=Y.shape[0]*5, repeat=False)
    # plt.show()
