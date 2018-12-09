import sys
sys.path.append('../../')

from libs.models.som_tensorflow import SOM
from libs.visualization.som.Grad_norm import SOM_Umatrix
from libs.datasets.artificial import animal


if __name__ == '__main__':
    

    title="animal map"
    umat_resolution = 100 #U-matrix表示の解像度

    # Data import : Animal Data
    X, labels = animal.load_data()

    

    # Computing with default parameters
    som = SOM(X.shape[1], X.shape[0])
    som.predict(X)

    # Extracting last BMU position and last sigma value for U-Matrix
    Z = som.historyZ[-1]
    sigma = som.historyS[-1]

    som_umatrix = SOM_Umatrix(X=X,
                              Z=Z,
                              sigma=sigma,
                              labels=labels,
                              title_text=title,
                              resolution=umat_resolution)
    som_umatrix.draw_umatrix()
