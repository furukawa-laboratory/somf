from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.spatial import distance

class TSOM2:

    #########################################################################################################################
    #
    #  TSOM Algorithm, Batch Implementation powered by Tensorflow
    #
    #  TODO :
    #
    #  - Meshgrid implementation on Tensorflow ?
    #  - Regroup the variable in one tensor instead of split ?
    #
    #########################################################################################################################

    def __init__(self, dim, number_vectors, latentdim = [2,2], epochs=10, n=[2, 2], m=[2, 2], sigma_min=[0.2, 0.2], sigma_max=[2.0, 2.0],
                 tau=[50, 50], init='rand'):
        """
        Initializing the SOM Network through a TensorFlow graph

        :param dim: Dimension of the Input Data, 3D Tensor
        :param number_vectors: Number of Vectors in the Input Data array. Array format.
        :param latentdim: Latent dimension for Z1 and Z2. Only 2 or 1. If 1 is specified, only n is used to make the
        grid. Default is [2, 2]
        :param epochs: Number of epochs for iteration. Default is 50
        :param n: Length of the visualization maps. Default is [10, 10]
        :param m: Width pf the visualization map. Default is [10, 10]
        :param signa_min: Minimum values for sigmas. Default is [0.3, 0.3]
        :param sigma_max: Maximum values for sigmas. Default is [2.2, 2.2]
        :param tau: Learning rates. Default is [50, 50]
        :param init: Z Initialization Method. Default is at Random. If shape, please provide an array of matrixes !

        Note: Preprocess the data array for it to have a 3rd dimension !

        """
        print('\nInitializing SOM Tensorflow Graph...\n')

        # Parameters initializing
        self.number_vectors = number_vectors
        self.dimension = dim
        self.n = n
        self.m = m
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.tau = tau
        self.epochs = epochs
        self.latentdim = latentdim

        # Setting the graph used by TensorFlow
        self.graph = tf.Graph()

        with self.graph.as_default():

           # Placeholder for the Input_data
           self.input_data = tf.placeholder(shape=[self.number_vectors[0], self.number_vectors[1], self.dimension],
                                             dtype=tf.float64, name='Input_Data')

           self.iter_no = tf.placeholder(dtype=tf.float64, name='Current_Iteration_Number')

           with tf.name_scope('Zeta_Matrixes'):

                if latentdim[0] == 2:
                    zeta1 = np.dstack(np.meshgrid(np.linspace(-1, 1, self.n[0], endpoint=True), np.linspace(-1, 1, self.m[0], endpoint=True))).reshape(
                        -1, latentdim[0])
                    self.K1 = self.n[0]*self.m[0]
                    self.Zeta1 = tf.constant(zeta1, dtype=tf.float64)
                else:
                    zeta1 = np.dstack(np.linspace(-1,1, self.n[0], endpoint=True)).reshape(-1, latentdim[0])
                    self.K1 = self.n[0]
                    self.Zeta1 = tf.constant(zeta1, dtype=tf.float64)

                if latentdim[1] == 2:
                    zeta2 = np.dstack(np.meshgrid(np.linspace(-1, 1, self.n[1], endpoint=True), np.linspace(-1, 1, self.m[1], endpoint=True))).reshape(
                        -1, latentdim[1])
                    self.K2 = self.n[1]*self.m[1]
                    self.Zeta2 = tf.constant(zeta2, dtype=tf.float64)
                else:
                    zeta2 = np.dstack(np.linspace(-1, 1, self.n[1], endpoint=True)).reshape(-1, latentdim[1])
                    self.K2 = self.n[1]
                    self.Zeta2 = tf.constant(zeta2, dtype=tf.float64)


           with tf.name_scope('Z'):

                if isinstance(init, str) and init == 'rand':

                    self.Z1 = tf.Variable(
                        tf.random_uniform(shape=[self.number_vectors[0], latentdim[0]], dtype=tf.float64) * 2.0 - 1.0)
                    self.Z2 = tf.Variable(
                        tf.random_uniform(shape=[self.number_vectors[1], latentdim[1]], dtype=tf.float64) * 2.0 - 1.0)

                elif isinstance(init, (tuple, list)) and init[0].shape == (self.number_vectors[0], latentdim[0]) and init[
                    1].shape == (self.number_vectors[1], latentdim[1]):

                    self.Z1 = tf.Variable(initial_value=init[0], dtype=tf.float64)
                    self.Z2 = tf.Variable(initial_value=init[1], dtype=tf.float64)

                else:
                    raise ValueError("invalid init: {}".format(init))

           # Weights vectors (Y), BMU and Vectors image in 2D (Z and Zeta), initialized at random
           with tf.name_scope('Weights_Tensors'):

                self.Y = tf.Variable(tf.random_normal(shape=[self.K1, self.K2, self.dimension],
                                             dtype=tf.float64))

           with tf.name_scope('Sigma'):

                # Variable to store sigma value
                self.sigma_value = tf.Variable(tf.zeros(shape=[2], dtype=tf.float64), name='Sigma_value')

                # Assign value of sigma depending on iteration number
                self.sigma_update = tf.assign(self.sigma_value, self.sigma(), name='Updating_Sigma_Value')

            ################################### COOPERATION AND ADAPTATION STEP ########################################

           # Compute & Update the new weights
           with tf.name_scope('Updating_Weights_Y'):
                self.dist = tf.pow(self.pairwise_dist(self.Zeta2, self.Z2), 2)
                self.R1, _ = self.neighboor_update(self.Zeta1, self.Z1, 0)
                self.R2, self.d2 = self.neighboor_update(self.Zeta2, self.Z2, 1)
                self.U = tf.einsum('lj,ijd->ild', self.R2, self.input_data)
                self.V = tf.einsum('ki,ijd->kjd', self.R1, self.input_data)
                
                self.train_update = tf.assign(self.Y, tf.einsum('ki,lj,ijd->kld', self.R1, self.R2, self.input_data))


            ########################################## COMPETITIVE STEP ################################################

           # Return a list with the number of each Best Best Matching Unit for each Input Vectors
           with tf.name_scope('Getting_BMU_Nodes'):

                self.bmu_nodes1 = tf.reshape(
                    self.winning_nodes(self.U[:, None, :, :], self.train_update[None, :, :, :], [2, 3]),
                    shape=[self.number_vectors[0]])
                self.bmu_nodes2 = tf.reshape(
                    self.winning_nodes(self.V[:, :, None, :], self.train_update[:, None, :, :], [0, 3]),
                    shape=[self.number_vectors[1]])

           # BMU Vectors extractions, each vector is a 2 dimension one (for mapping)
           with tf.name_scope('Updating_Z'):
                self.Z1_update = tf.assign(self.Z1,
                                           tf.reshape(tf.gather(self.Zeta1, self.bmu_nodes1,
                                                                name='Choosing_Zeta_Based_On_BMU1'),
                                                      shape=[self.number_vectors[0], latentdim[0]]))
                self.Z2_update = tf.assign(self.Z2,
                                           tf.reshape(tf.gather(self.Zeta2, self.bmu_nodes2,
                                                                name='Choosing_Zeta_Based_On_BMU2'),
                                                      shape=[self.number_vectors[1], latentdim[1]]))
            # Initializing Session and Variable
           self.session = tf.Session()
           self.session.run(tf.global_variables_initializer())

           print('\nReady !\n')

    def neighboor_update(self, Zeta, Z, k):
        """
        Computing the new weights for the Weights Tensor (Y)
        See Batch SOM Algorithm for details

        :returns: A Tensor of shape (Number_of_Reference_Vectors, Dimension)
        """

        with tf.name_scope('New_Weights_Computation'):
            # Matrix computing distance between each reference vectors and the Best Matching Unit
            # Shape : Number_of_Reference_Vectors x Number_of_Input_Data_Vectors
            with tf.name_scope('Neighboors_Distance_Matrix'):
                H = tf.pow(self.pairwise_dist(Zeta, Z), 2)

            # Matrix computing the neighboorhood based on the distance Matrix H for each Reference Vectors
            # Shape : Number_of_Reference_Vectors x Number_of_Input_Data_Vectors
            with tf.name_scope('Determining_Neighboorhood'):
                G = tf.exp(-H / (2 * tf.pow(tf.gather_nd(self.sigma_update, [k]), 2)))


            # Computing invert Matrix of Sum
            # Shape : Number_of_Reference_Vectors x 1
            with tf.name_scope('Computing_Invert_Sum_Distance'):
                L = tf.expand_dims(tf.reduce_sum(G, axis=1), 1)
                Linv = tf.convert_to_tensor(np.reciprocal(L))

            # Matrix computing the sum between the G Matrix and the invertMatrix
            # Shape : Number_of_Reference_Vectors x Number_of_Input_Vectors
            with tf.name_scope('Sum_betwwen_Neighboorhood_Matrix_and_Invert_Sum'):
                return G * Linv, G

    def sigma(self):
        """
        Computing the evolution of sigma based on the iteration number
        Wrapping the function max using py_func to get a tensor

        :returns: The value of sigma for this iteration
        """

        with tf.name_scope('Sigma_computation'):
            return tf.maximum(self.sigma_min * tf.constant(1, tf.float64), self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - (self.iter_no / self.tau)))

    def winning_nodes(self, M, Y, axis):
        """
        Return the list of the Best Matching Units by computing the distance between
        the Weights Tensor and the Input Data Tensor

        :returns: A Tensor of shape (Number_of_Input_vectors, 1) containing the list of
        Best Matching Unit
        """

        with tf.name_scope('Winning_node'):
            dist = tf.square(M - Y)

            self.sumR = tf.reduce_sum(dist, axis=axis)

            arg = tf.argmin(self.sumR, 1, name='Argmin')

            return arg

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

        #return tf.reshape(tf.py_func(distance.cdist, [A, B], tf.float64), shape=[D.get_shape()[0], D.get_shape()[1]])

    def predict(self, data, graph=False):
        """
        Launch prediction on data

        :param data: An array of data to predict on
        :param graph: Default is False. Save the graph to be visualised in tensorboard (default directory 'output'
        will be at the same location of the file using the algorithm.
        """
        print('\nPredicting out of {0} epochs\n'.format(self.epochs))
        bar = tqdm(range(self.epochs))

        # History saving for BMU, Weights and sigma value
        self.historyZ1 = np.zeros((self.epochs, self.number_vectors[0], self.latentdim[0]))
        self.historyZ2 = np.zeros((self.epochs, self.number_vectors[1], self.latentdim[1]))
        self.historyY = np.zeros((self.epochs, self.K1, self.K2, self.dimension))
        self.historyS = np.zeros((self.epochs, 2))
        self.historyB1 = np.zeros(self.number_vectors[0])
        self.historyB2 = np.zeros(self.number_vectors[1])
        self.histDist = np.zeros((self.epochs, self.number_vectors[1], self.K2))

        for i in bar:
            # Computing each iteration for the whole batch (ie : Update the weights each iteration) + Saving history
            self.historyS[i], self.historyY[i, :, :], self.historyZ1[i, :], self.historyZ2[i, :], self.historyB1, self.historyB2,self.histDist[i] = self.session.run(
                [self.sigma_update, self.train_update, self.Z1_update, self.Z2_update, self.bmu_nodes1, self.bmu_nodes2,self.sumR],
                feed_dict={self.input_data: data, self.iter_no: i})


        if graph == True:
            writer = tf.summary.FileWriter('output', self.session.graph)
            writer.close()

        print('\nClosing Tensorflow Session...\n')

        # Closing tf session
        self.session.close()
