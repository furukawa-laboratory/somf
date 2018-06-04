import numpy as np
import os


def load_data(retlabel=True, mean_zero=True):
    dirname = '3PhData'
    filename = 'DataTrn.txt'
    filepath = os.path.join(os.path.dirname(__file__), dirname, filename)

    x = np.loadtxt(filepath)
    if mean_zero:
        x -= x.mean(0)

    if retlabel:
        labelfilename = 'DataTrnLbls.txt'
        labelfilepath = os.path.join(os.path.dirname(__file__), dirname, labelfilename)
        label = np.loadtxt(labelfilepath).nonzero()[1]
        return x, label
    else:
        return x


