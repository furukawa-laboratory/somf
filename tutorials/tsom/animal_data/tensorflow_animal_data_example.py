from libs.datasets.artificial.animal import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V
from libs.models.tsom_tensorflow import TSOM2

if __name__ == '__main__':
    X, labels_animal, labels_feature = load_data(retlabel_animal=True, retlabel_feature=True)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    n=[10, 10]
    m=[10, 10]


    tsom = TSOM2(X.shape[2], [X.shape[0], X.shape[1]], epochs=50, n=n, m=m, sigma_max=[2.0, 2.0], sigma_min=[0.2, 0.2], tau=[50,50])
    tsom.predict(X)






    comp = TSOM2_V(y=tsom.historyY[-1], winner1=tsom.bmus1[-1], winner2=tsom.bmus2[-1], label1=labels_animal,
                   label2=labels_feature)
    comp.draw_map()