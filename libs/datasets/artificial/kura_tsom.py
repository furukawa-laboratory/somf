import numpy as np
import random
import itertools

def load_kura_tsom(xsamples, ysamples, missing_rate=None,retz=False):
    z1 = np.linspace(-1, 1, xsamples)
    z2 = np.linspace(-1, 1, ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2, indexing='ij')
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = z1_repeated ** 2.0 - z2_repeated ** 2.0

    x = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis], x3[:, :, np.newaxis]), axis=2)
    truez = np.concatenate((z1_repeated[:, :, np.newaxis], z2_repeated[:, :, np.newaxis]), axis=2)

    #欠損値を入れない場合(missing_rateが0か特に指定していない場合はそのまま返す)
    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x, truez
        else:
            return x

    #欠損値を入れる場合
    else:
        # データの欠損数を決定する
        if 0 < missing_rate < 1:  #全体のサンプル数から率から欠損数を計算する
            missing_num = missing_rate
            all_samples = xsamples * ysamples
            missing_rate = int(all_samples * missing_num)
        else:  #missing_rateが0~1でなかったら場合はerror文を返す
            raise ValueError("invalid missing_rate: {}\nmissing_rate is 0<=missing_rate<1.".format(missing_rate))

        #どのデータを欠損させるかを決定する
        
        # list1とlist2の全組み合わせの配列を作成して、それをシャッフルして0番目からmissing_num個だけ欠損させる
        missing_list1 = np.arange(xsamples)
        missing_list2 = np.arange(ysamples)
        p = list(itertools.product(missing_list1, missing_list2))  # List数はN1*N2
        random.shuffle(p)  # listをshuffle

        Gamma = np.ones((xsamples, ysamples))#Gammaはどのデータが欠損かを表す

        for n in np.arange(missing_rate):  # 欠損させたいデータ数分、Gammaの要素を欠損させる
            tempp = p[n]
            i = tempp[0]
            j = tempp[1]
            if Gamma[i, j] == 1:
                Gamma[i, j] = 0
            elif Gamma[i, j] == 0:#同じ場所を欠損させようとしたらエラーを吐く
                raise ValueError("invalid Gamma: {}\n".format(Gamma))

        #true_zを欠損させる
        for n in np.arange(missing_rate):
            tempp = p[n]
            i = tempp[0]
            j = tempp[1]
            truez[i,j,:]=np.nan

        #Gammaに基づいてデータ行列を欠損させる
        # 欠損値をNan埋めする
        for i in np.arange(xsamples):
            for j in np.arange(ysamples):
                if Gamma[i, j] == 0:
                    x[i, j, :] = np.nan

        if retz:
            return x,truez,Gamma
        else:
            return x,Gamma




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 10
    ysamples = 10

    #欠損なしver
    # x, truez = load_kura_tsom(xsamples, ysamples, retz=True)

    # 欠損ありver
    x, truez, Gamma = load_kura_tsom(xsamples, ysamples, retz=True,missing_rate=0.7)

    fig = plt.figure(figsize=[10, 5])
    ax_x = fig.add_subplot(1, 2, 1, projection='3d')
    ax_truez = fig.add_subplot(1, 2, 2)
    ax_x.scatter(x[:, :, 0].flatten(), x[:, :, 1].flatten(), x[:, :, 2].flatten(), c=x[:, :, 0].flatten())
    ax_truez.scatter(truez[:, :, 0].flatten(), truez[:, :, 1].flatten(), c=x[:, :, 0].flatten())
    ax_x.set_title('Generated three-dimensional data')
    ax_truez.set_title('True two-dimensional latent variable')
    plt.show()

