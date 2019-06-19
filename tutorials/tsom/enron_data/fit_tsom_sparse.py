import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from libs.models.tsom_sparse import TSOM2
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer


def main():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz'

    # read data
    print('read data')
    df = pd.read_csv(url, skiprows=3, header=None, sep=' ', names=['doc_id', 'vocab_id', 'freq'])

    # create sparse data
    doc_c = CategoricalDtype(sorted(df.doc_id.unique()), ordered=True)
    vocab_c = CategoricalDtype(sorted(df.vocab_id.unique()), ordered=True)
    row = df.doc_id.astype(doc_c).cat.codes
    col = df.vocab_id.astype(vocab_c).cat.codes
    data = csr_matrix((df.freq, (row, col)), shape=(doc_c.categories.size, vocab_c.categories.size))

    # set parameter
    latent_dim = 2
    resolution = 10
    sigma_max = 2.0
    sigma_min = 0.2
    tau = 20.0
    nb_epoch = 100
    is_direct = True

    # learn
    print('learn')
    tsom2 = TSOM2(data, latent_dim, resolution, sigma_max, sigma_min, tau)
    tsom2.fit(nb_epoch, is_direct)

    # draw
    viewer = TSOM2_Viewer(tsom2.y, tsom2.k_star1, tsom2.k_star2)
    viewer.draw_map()


if __name__ == '__main__':
    main()