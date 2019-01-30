# coding=utf-8
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..tools.create_zeta import create_zeta


class TSOM2():
    def __init__(self, N1, N2, observed_dim, latent_dim, epochs, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init='random'):

        self.N1 = N1
        self.N2 = N2
        self.observed_dim = observed_dim

        self.epochs = epochs

        # 最大近傍半径(SIGMAX)の設定
        if type(SIGMA_MAX) is float:
            self.SIGMA1_MAX = SIGMA_MAX
            self.SIGMA2_MAX = SIGMA_MAX
        elif isinstance(SIGMA_MAX, (list, tuple)):
            self.SIGMA1_MAX = SIGMA_MAX[0]
            self.SIGMA2_MAX = SIGMA_MAX[1]
        else:
            raise ValueError("invalid SIGMA_MAX: {}".format(SIGMA_MAX))

        # 最小近傍半径(SIGMA_MIN)の設定
        if type(SIGMA_MIN) is float:
            self.SIGMA1_MIN = SIGMA_MIN
            self.SIGMA2_MIN = SIGMA_MIN
        elif isinstance(SIGMA_MIN, (list, tuple)):
            self.SIGMA1_MIN = SIGMA_MIN[0]
            self.SIGMA2_MIN = SIGMA_MIN[1]
        else:
            raise ValueError("invalid SIGMA_MIN: {}".format(SIGMA_MIN))

        # 時定数(TAU)の設定
        if type(TAU) is int:
            self.TAU1 = TAU
            self.TAU2 = TAU
        elif isinstance(TAU, (list, tuple)):
            self.TAU1 = TAU[0]
            self.TAU2 = TAU[1]
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

        # resolutionの設定
        if type(resolution) is int:
            resolution1 = resolution
            resolution2 = resolution
        elif isinstance(resolution, (list, tuple)):
            resolution1 = resolution[0]
            resolution2 = resolution[1]
        else:
            raise ValueError("invalid resolution: {}".format(resolution))

        self.K1 = resolution1 * resolution1
        self.K2 = resolution2 * resolution2

        # 潜在空間の設定
        if type(latent_dim) is int:  # latent_dimがintであればどちらのモードも潜在空間の次元は同じ
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim

        elif isinstance(latent_dim, (list, tuple)):
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))
            # latent_dimがlist,float,3次元以上はエラー

        # Setting the graph used by TensorFlow
        self.graph = tf.Graph()

        with self.graph.as_default():
            # Placeholder for the Input_data
            self.input_data = tf.placeholder(shape=[self.N1, self.N2, self.observed_dim], dtype=tf.float64,
                                             name='Input_Data')

            self.iter_no = tf.placeholder(dtype=tf.float64, name='Current_Iteration_Number')

            # Weights vectors (Y), BMU and Vectors image in 2D (Z and Zeta), initialized at random
            with tf.name_scope('Zeta_Matrix'):
                self.Zeta1 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim1, resolution=resolution1,
                                         include_min_max=True)
                self.Zeta2 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim2, resolution=resolution2,
                                         include_min_max=True)

            if isinstance(init, str) and init == 'rand':
                with tf.name_scope('Z'):
                    self.Z1 = tf.Variable(
                        tf.random_uniform(shape=[self.N1, self.latent_dim1], dtype=tf.float64) * 2.0 - 1.0)
                    self.Z2 = tf.Variable(
                        tf.random_uniform(shape=[self.N2, self.latent_dim2], dtype=tf.float64) * 2.0 - 1.0)

            elif isinstance(init, (tuple, list)) and len(init) == 2:
                if isinstance(init[0], np.ndarray) and init[0].shape == (self.N1, self.latent_dim1):
                    with tf.name_scope('Z'):
                        self.Z1 = tf.Variable(initial_value=init[0].copy(), dtype=tf.float64)
                else:
                    raise ValueError("invalid inits[0]: {}".format(init))
                if isinstance(init[1], np.ndarray) and init[1].shape == (self.N2, self.latent_dim2):
                    with tf.name_scope('Z'):
                        self.Z2 = tf.Variable(initial_value=init[1].copy(), dtype=tf.float64)
                else:
                    raise ValueError("invalid inits[1]: {}".format(init))
            else:
                raise ValueError("invalid inits: {}".format(init))

            with tf.name_scope('Sigma'):
                # Variable to store sigma value
                self.sigma1_value = tf.Variable(tf.zeros(shape=(), dtype=tf.float64), name='Sigma1_value')
                self.sigma2_value = tf.Variable(tf.zeros(shape=(), dtype=tf.float64), name='Sigma2_value')

                # Assign value of sigma depending on iteration number
                self.sigma1_update = tf.assign(self.sigma1_value, self.sigma(), name='Updating_Sigma1_Value')
                self.sigma2_update = tf.assign(self.sigma2_value, self.sigma(), name='Updating_Sigma2_Value')

            with tf.name_scope('Weights_Tensor'):
                self.U = tf.Variable(tf.random_normal(shape=[self.N1, self.K2, self.observed_dim], dtype=tf.float64))
                self.V = tf.Variable(tf.random_normal(shape=[self.K1, self.N2, self.observed_dim], dtype=tf.float64))
                self.Y = tf.Variable(tf.random_normal(shape=[self.K1, self.K2, self.observed_dim], dtype=tf.float64))

            ################################### COOPERATION AND ADAPTATION STEP ########################################

            # Compute & Update the new weights
            with tf.name_scope('Updating_Weights'):
                self.train_update_U = tf.assign(self.U, self.neighboor_update()[0])
                self.train_update_V = tf.assign(self.V, self.neighboor_update()[1])
                self.train_update_Y = tf.assign(self.Y, self.neighboor_update()[2])

            ########################################## COMPETITIVE STEP ################################################

            # Return a list with the number of each Best Best Matching Unit for each Input Vectors
            with tf.name_scope('Getting_BMU_Nodes'):
                self.bmu1_nodes = tf.reshape(self.winning_nodes()[0], shape=[self.N1])
                self.bmu2_nodes = tf.reshape(self.winning_nodes()[1], shape=[self.N2])

            # BMU Vectors extractions, each vector is a 2 dimension one (for mapping)
            with tf.name_scope('Updating_Z'):
                self.Z1_update = tf.assign(self.Z1,
                                           tf.reshape(
                                               tf.gather(self.Zeta1, self.bmu1_nodes,
                                                         name='Choosing_Zeta_Based_On_BMU'),
                                               shape=[self.N1, 2]))
                self.Z2_update = tf.assign(self.Z2,
                                           tf.reshape(
                                               tf.gather(self.Zeta2, self.bmu2_nodes,
                                                         name='Choosing_Zeta_Based_On_BMU'),
                                               shape=[self.N2, 2]))

            # History saving for BMU, Weights and sigma value
            self.historyZ1 = np.zeros((epochs, self.N1, self.latent_dim1))
            self.historyZ2 = np.zeros((epochs, self.N2, self.latent_dim2))
            self.historyY = np.zeros((epochs, self.K1, self.K2, self.observed_dim))
            self.historyS1 = np.zeros(epochs)
            self.historyS2 = np.zeros(epochs)

            # Initializing Session and Variable
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

            print('\nReady !\n')

    def neighboor_update(self):
        with tf.name_scope('New_Weights_Computation'):
            # Matrix computing distance between each reference vectors and the Best Matching Unit
            with tf.name_scope('Neighboors_Distance_Matrix'):
                H1 = tf.reshape(tf.pow(self.pairwise_dist(self.Zeta1, self.Z1), 2), shape=[self.K1, self.N1])
                H2 = tf.reshape(tf.pow(self.pairwise_dist(self.Zeta2, self.Z2), 2), shape=[self.K2, self.N2])

                # Matrix computing the neighboorhood based on the distance Matrix H for each Reference Vectors
            with tf.name_scope('Determining_Neighboorhood'):
                G1 = tf.reshape(tf.exp(-H1 / (2 * tf.pow(self.sigma1_update, 2))), shape=[self.K1, self.N1])
                G2 = tf.reshape(tf.exp(-H2 / (2 * tf.pow(self.sigma2_update, 2))), shape=[self.K2, self.N2])

            # Computing invert Matrix of Sum
            with tf.name_scope('Computing_Invert_Sum_Distance'):
                L1 = tf.expand_dims(tf.reduce_sum(G1, axis=1), 1)
                L1inv = tf.reciprocal(L1)
                L2 = tf.expand_dims(tf.reduce_sum(G2, axis=1), 1)
                L2inv = tf.reciprocal(L2)

            # Matrix computing the sum between the G Matrix and the invertMatrix
            with tf.name_scope('Sum_betwwen_Neighboorhood_Matrix_and_Invert_Sum'):
                R1 = G1 * L1inv
                R2 = G2 * L2inv

            # Computing the weights
            with tf.name_scope('Computing_Weights'):
                U = np.einsum('lj,ijd->ild', R2, self.input_data)
                V = np.einsum('ki,ijd->kjd', R1, self.input_data)
                Y = np.einsum('ki,lj,ijd->kld', R1, R2, self.input_data)
                return U, V, Y

    def sigma(self):
        with tf.name_scope('Sigma_computation'):
            sigma1 = tf.maximum(tf.cast(self.SIGMA1_MIN, tf.float64),
                                self.SIGMA1_MAX * (1 - (self.iter_no / self.TAU1)))
            sigma2 = tf.maximum(tf.cast(self.SIGMA2_MIN, tf.float64),
                                self.SIGMA2_MAX * (1 - (self.iter_no / self.TAU2)))
            return sigma1, sigma2

    def winning_nodes(self):
        with tf.name_scope('Winning_node'):
            self.bmu1_nodes = tf.argmin(
                tf.reduce_sum(tf.square(self.train_update_U[:, None, :, :] - self.train_update_Y[None, :, :, :]),
                              axis=(2, 3)), axis=1)
            self.bmu2_nodes = tf.argmin(
                tf.reduce_sum(tf.square(self.train_update_V[:, :, None, :] - self.train_update_Y[:, None, :, :]),
                              axis=(0, 3)), axis=1)
            return self.bmu1_nodes, self.bmu2_nodes

    def pairwise_dist(self, A, B):
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          A,    [m,d] matrix
          B,    [n,d] matrix
        Returns:
          D,    [m,n] matrix of pairwise distances

        Credits : https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
        """
        with tf.name_scope('Pairwise_Distance'):
            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(A), 1)
            nb = tf.reduce_sum(tf.square(B), 1)

            # na as a row and nb as a co"lumn vectors
            na = tf.reshape(na, [-1, 1])
            nb = tf.reshape(nb, [1, -1])

            # return pairwise euclidead difference matrix
            D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
        return D

    def predict(self, data, graph=False):
        """
        Launch prediction on data

        :param data: An array of data to predict on
        :param graph: Default is False. Save the graph to be visualised in tensorboard (default directory 'output'
        will be at the same location of the file using the algorith;.
        """
        print('\nPredicting out of {0} epochs\n'.format(self.epochs))
        bar = tqdm(range(self.epochs))
        for i in bar:
            # Computing each iteration for the whole batch (ie : Update the weights each iteration) + Saving history
            self.historyS1[i], self.historyS2[i], self.historyY[i], self.historyZ1[i], self.historyZ2[
                i] = self.session.run(
                [self.sigma1_update, self.sigma2_update, self.train_update_Y, self.Z1_update, self.Z2_update],
                feed_dict={self.input_data: data, self.iter_no: i})

        if graph == True:
            writer = tf.summary.FileWriter('output', self.session.graph)
            writer.close()

        print('\nClosing Tensorflow Session...\n')

        # Closing tf session
        self.session.close()
