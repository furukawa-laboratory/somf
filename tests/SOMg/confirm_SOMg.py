import numpy as np
from libs.models.som import SOM

K_MAX = 8


class SOMg(object):
    # SQUARE 限定
    def __init__(self, som: SOM):
        self.som = som
        self.resolution = int(pow(self.som.K, 1 / self.som.L))
        self.umat = np.zeros((som.K, K_MAX + 1))
        self.umat_color = np.zeros((som.K, K_MAX + 1))
        self.umat_matrix = np.zeros((self.resolution, 4, self.resolution, 4))

    def plot_umatrix(self, axes, func):
        self.__calc_umatrix(func)
        self.__draw_umatrix(axes)

    def __calc_umatrix_color(self, n, k):
        if k == K_MAX:
            aver = self.umat[n, k]
        else:
            ne = self.__neighbor_unit(n, k)
            if k == 0:
                aver = self.umat[n, 7] / 2 + \
                       self.umat[n, 0] * 2 + \
                       self.umat[n, 1] / 2 + \
                       self.umat[n, 6] + \
                       self.umat[n, 8] + \
                       self.umat[n, 2]
                nn = 6
                if ne >= 0:
                    aver += self.umat[ne, 5] / 2 + \
                            self.umat[ne, 3] / 2 + \
                            self.umat[ne, 6] + \
                            self.umat[ne, 8] + \
                            self.umat[ne, 2]
                    nn = 10
            elif k == 1:
                if ne < 0:
                    aver = self.umat[n, 0] + \
                           self.umat[n, 1] * 2 + \
                           self.umat[n, 2] + \
                           self.umat[n, 8]
                    nn = 5
                else:
                    aver = self.umat[n, 0] + \
                           self.umat[n, 1] + \
                           self.umat[n, 2] + \
                           self.umat[n, 8] + \
                           self.umat[ne, 4] + \
                           self.umat[ne, 6] + \
                           self.umat[ne, 8]
                    n1 = self.__neighbor_unit(n, 0)
                    aver += self.umat[n1, 3] + self.umat[n1, 8]
                    n2 = self.__neighbor_unit(n, 2)
                    aver += self.umat[n2, 8]
                    nn = 10
            elif k == 2:
                aver = self.umat[n, 1] / 2 + \
                       self.umat[n, 2] * 2 + \
                       self.umat[n, 3] / 2 + \
                       self.umat[n, 0] + \
                       self.umat[n, 8] + \
                       self.umat[n, 4]
                nn = 6
                if ne >= 0:
                    aver += self.umat[ne, 5] / 2 + \
                            self.umat[ne, 7] / 2 + \
                            self.umat[ne, 0] + \
                            self.umat[ne, 8] + \
                            self.umat[ne, 4]
                    nn = 10
            elif k == 3:
                if ne < 0:
                    aver = self.umat[n, 2] + \
                           self.umat[n, 3] * 2 + \
                           self.umat[n, 4] + \
                           self.umat[n, 8]
                    nn = 5
                else:
                    aver = self.umat[n, 2] + \
                           self.umat[n, 3] + \
                           self.umat[n, 4] + \
                           self.umat[n, 8] + \
                           self.umat[ne, 6] + \
                           self.umat[ne, 0] + \
                           self.umat[ne, 8]
                    n1 = self.__neighbor_unit(n, 2)
                    aver += self.umat[n1, 5] + self.umat[n1, 8]
                    n2 = self.__neighbor_unit(n, 4)
                    aver += self.umat[n2, 8]
                    nn = 10
            elif k == 4:
                aver = self.umat[n, 3] / 2 + \
                       self.umat[n, 4] * 2 + \
                       self.umat[n, 5] / 2 + \
                       self.umat[n, 2] + \
                       self.umat[n, 8] + \
                       self.umat[n, 6]
                nn = 6
                if ne >= 0:
                    aver += self.umat[ne, 1] / 2 + \
                            self.umat[ne, 7] / 2 + \
                            self.umat[ne, 2] + \
                            self.umat[ne, 8] + \
                            self.umat[ne, 6]
                    nn = 10
            elif k == 5:
                if ne < 0:
                    aver = self.umat[n, 4] + \
                           self.umat[n, 5] * 2 + \
                           self.umat[n, 6] + \
                           self.umat[n, 8]
                    nn = 5
                else:
                    aver = self.umat[n, 4] + \
                           self.umat[n, 5] + \
                           self.umat[n, 6] + \
                           self.umat[n, 8] + \
                           self.umat[ne, 0] + \
                           self.umat[ne, 2] + \
                           self.umat[ne, 8]
                    n1 = self.__neighbor_unit(n, 4)
                    aver += self.umat[n1, 7] + self.umat[n1, 8]
                    n2 = self.__neighbor_unit(n, 6)
                    aver += self.umat[n2, 8]
                    nn = 10
            elif k == 6:
                aver = self.umat[n, 5] / 2 + \
                       self.umat[n, 6] * 2 + \
                       self.umat[n, 7] / 2 + \
                       self.umat[n, 4] + \
                       self.umat[n, 8] + \
                       self.umat[n, 0]
                nn = 6
                if ne >= 0:
                    aver += self.umat[ne, 1] / 2 + \
                            self.umat[ne, 3] / 2 + \
                            self.umat[ne, 4] + \
                            self.umat[ne, 8] + \
                            self.umat[ne, 0]
                    nn = 10
            elif k == 7:
                if ne < 0:
                    aver = self.umat[n, 6] + \
                           self.umat[n, 7] * 2 + \
                           self.umat[n, 0] + \
                           self.umat[n, 8]
                    nn = 5
                else:
                    aver = self.umat[n, 6] + \
                           self.umat[n, 7] + \
                           self.umat[n, 0] + \
                           self.umat[n, 8] + \
                           self.umat[ne, 2] + \
                           self.umat[ne, 4] + \
                           self.umat[ne, 8]
                    n1 = self.__neighbor_unit(n, 6)
                    aver += self.umat[n1, 1] + self.umat[n1, 8]
                    n2 = self.__neighbor_unit(n, 0)
                    aver += self.umat[n2, 8]
                    nn = 10
            else:
                raise ValueError()
            aver /= nn

        scale = (aver - self.mu) / (self.sigma * 4.0) + 0.5

        return np.clip(scale, 0, 1)

    def __draw_umatrix(self, axes):
        for i in range(self.som.K):
            for k in range(K_MAX + 1):
                self.umat_color[i, k] = self.__calc_umatrix_color(i, k)
            cell_image = np.zeros(16)
            cell_image[0] = self.umat_color[i, 7]
            cell_image[1] = self.umat_color[i, 0]
            cell_image[2] = self.umat_color[i, 0]
            cell_image[3] = self.umat_color[i, 1]
            cell_image[4] = self.umat_color[i, 6]
            cell_image[5] = self.umat_color[i, 8]
            cell_image[6] = self.umat_color[i, 8]
            cell_image[7] = self.umat_color[i, 2]
            cell_image[8] = self.umat_color[i, 6]
            cell_image[9] = self.umat_color[i, 8]
            cell_image[10] = self.umat_color[i, 8]
            cell_image[11] = self.umat_color[i, 2]
            cell_image[12] = self.umat_color[i, 5]
            cell_image[13] = self.umat_color[i, 4]
            cell_image[14] = self.umat_color[i, 4]
            cell_image[15] = self.umat_color[i, 3]

            ix = i % self.resolution
            iy = i // self.resolution
            self.umat_matrix[iy, :, ix, :] = cell_image.reshape(4, 4)

        axes.imshow(self.umat_matrix.reshape(self.resolution * 4, self.resolution * 4), vmin=0, vmax=1, cmap='jet')

    def __calc_umatrix(self, func):
        sum = 0.0
        sum2 = 0.0
        count = 0
        for i1 in range(self.som.K):
            neighbor_total_distance = 0.0
            neighbor_count = 0
            for k in range(K_MAX):
                i2 = self.__neighbor_unit(i1, k)
                if i2 >= 0 and i2 < self.som.K:
                    distance = func(i1, i2)
                    self.umat[i1, k] = distance
                    neighbor_total_distance += distance
                    neighbor_count += 1
                    sum += distance
                    sum2 += distance ** 2
                    count += 1
            self.umat[i1, K_MAX] = neighbor_total_distance / neighbor_count
        self.mu = sum / count
        self.sigma = np.sqrt(sum2 / count - self.mu ** 2)
        for i1 in range(self.som.K):
            for k in range(K_MAX):
                i2 = self.__neighbor_unit(i1, k)
                if i2 < 0:
                    self.umat[i1, k] = self.umat[i1, K_MAX]
        self.umat=np.loadtxt("Umatrix_color_scale.txt")

    def __neighbor_unit(self, i, k):
        self.resolution = int(pow(self.som.K, 1 / self.som.L))
        ix = i % self.resolution
        iy = i // self.resolution
        if k == 0:
            jy = iy - 1
            jx = ix
        elif k == 1:
            jy = iy - 1
            jx = ix + 1
        elif k == 2:
            jy = iy
            jx = ix + 1
        elif k == 3:
            jy = iy + 1
            jx = ix + 1
        elif k == 4:
            jy = iy + 1
            jx = ix
        elif k == 5:
            jy = iy + 1
            jx = ix - 1
        elif k == 6:
            jy = iy
            jx = ix - 1
        elif k == 7:
            jy = iy - 1
            jx = ix - 1
        else:
            raise ValueError()
        if jx < 0 or jx >= self.resolution:
            return -1
        if jy < 0 or jy >= self.resolution:
            return -1

        return jy * self.resolution + jx
