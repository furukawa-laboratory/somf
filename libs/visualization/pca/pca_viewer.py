import plotly
import plotly.graph_objs as go
import numpy as np
from sklearn import decomposition


class PCA_VW:
    def __init__(self, x=None, dim=0, label=None):
        self.X = x
        self.dim = dim
        self.label = label

    def pca(self):
        if self.dim == 3:
            pca_3 = decomposition.PCA(n_components=3)
            x_pca = pca_3.fit_transform(self.X)
            trace = go.Scatter3d(x=x_pca[:, 0], y=x_pca[:, 1], z=x_pca[:, 2], mode='markers', marker=dict(line=dict(color='black', width=1)))
            layout = go.Layout(scene=dict(
                        xaxis=dict(ticks='', showticklabels=True),
                        yaxis=dict(ticks='', showticklabels=True),
                        zaxis=dict(ticks='', showticklabels=True),))
            fig = go.Figure(data=[trace], layout=layout)
            plotly.offline.plot(fig)
        elif self.dim == 2:
            pca_2 = decomposition.PCA(n_components=2)
            x_pca = pca_2.fit_transform(self.X)
            trace = go.Scatter(x=x_pca[:, 0], y=x_pca[:, 1], mode='markers', marker=dict(line=dict(color='black', width=1)))
            layout = go.Layout(scene=dict(xaxis=dict(ticks='', showticklabels=True),
                                          yaxis=dict(ticks='', showticklabels=True),))
            fig = go.Figure(data=[trace], layout=layout)
            plotly.offline.plot(fig)
        else:
            mn = np.mean(self.X, axis=0)
            z = self.X - mn
            cv = np.cov(z[:, 0], z[:, 1], bias=True)
            W, v = np.linalg.eig(cv)
            return W, v

