import numpy as np
import scipy.spatial.distance as dist
from tqdm import tqdm


class CCASOM:
    def __init__(self, x=0, kx=20, ky=20, sigma_max=2.0, sigma_min=0.2, tau=50, epoch=500, bmu=None, mode='coclustring'):
        # データ形式のチェック
        try:
            if x == 0:
                raise ValueError('データがセットされてないよ！！')
            if len(x) <= 1:
                raise ValueError('ビューが一個しかないよ！！')
            for i in range(1, len(x)):
                if x[i].shape[0] != x[i-1].shape[0]:
                    raise ValueError('ビュー１とビュー２のデータ数が合わないよ！確認して！')
        except NameError:
            raise

        # データとパラメータのセット
        self.X = x
        self.V = len(x)
        self.N = x[0].shape[0]
        # self.D = np.array([x1.shape[1], x2.shape[1]])
        self.KX = kx
        self.KY = ky
        self.K = kx * ky
        self.Sigma_max = sigma_max
        self.Sigma_min = sigma_min
        self.Tau = tau
        self.Epoch = epoch

        # CCASOMの初期化
        self.Y = []
        self.Metric = []
        for v in range(self.V):
            self.Y.append(np.zeros([self.Epoch, self.K, self.X[v].shape[1]]))
            self.Metric.append(np.zeros([self.Epoch, self.X[v].shape[1], self.X[v].shape[1]]))
        self.BMU = np.zeros([self.Epoch, self.N], dtype=int)
        self.Zeta = np.zeros([self.K, 2])
        zx = np.linspace(-1, 1, kx)
        zy = np.linspace(-1, 1, ky)
        zeta1, zeta2 = np.meshgrid(zx, zy)
        self.Zeta = np.c_[zeta1.ravel(), zeta2.ravel()]
        self.Distance = np.zeros([self.V, self.N, self.K])
        self.Mode = mode

        if bmu is None:
            self.BMU[0] = np.random.randint(0, self.K, size=self.N)
        else:
            self.BMU[0] = bmu

    # 学習
    def fit(self):
        self.__m_step(0)
        for v in range(self.V):
            self.Metric[v][0] = np.eye(self.X[v].shape[1])

        for t in tqdm(range(1, self.Epoch)):
            # SOMの学習
            self.__e_step_allview(t)
            self.__m_step(t)

            # メトリック学習
            if self.Mode == 'coclustring':
                for v in range(self.V):
                    v_teach_bmu = self.__calc_teach_bmu_other(t, v)
                    self.__calc_metric(t, v, v_teach_bmu)
            else:
                v_teach_bmu = self.__calc_teach_bmu_all(t)
                for v in range(self.V):
                    self.__calc_metric(t, v, v_teach_bmu)

    # 内部の関数
    def __e_step_allview(self, t):
        dis_xy = np.zeros([self.N, self.K])
        for v in range(self.V):
            A = self.__decomp_metric(self.Metric[v][t-1])
            x_met = self.X[v] @ A
            y_met = self.Y[v][t-1] @ A
            dis_xy = dis_xy + dist.cdist(x_met, y_met)
        self.BMU[t] = np.argmin(dis_xy, axis=1)

    def __m_step(self, t):
        sigma = self.Sigma_min + (self.Sigma_max - self.Sigma_min) * np.exp(-t / self.Tau)
        dis_zz = dist.cdist(self.Zeta, self.Zeta[self.BMU[t]], 'sqeuclidean')
        r = np.exp(-dis_zz / (2 * sigma * sigma))
        g = np.sum(r, axis=1)
        for v in range(self.V):
            self.Y[v][t] = (r.T / g).T @ self.X[v]

    # メトリック推定用のBMU推定
    def __calc_teach_bmu_other(self, t, v_minus):
        dis_xy = np.zeros([self.N, self.K])
        loop_v = np.arange(self.V)
        np.delete(loop_v, v_minus)
        for v in range(self.V):
            A = self.__decomp_metric(self.Metric[v][t-1])
            x_met = self.X[v] @ A
            y_met = self.Y[v][t] @ A
            dis_xy = dis_xy + dist.cdist(x_met, y_met)
        winner = np.argmin(dis_xy, axis=1)
        return winner

    def __calc_teach_bmu_all(self, t):
        dis_xy = np.zeros([self.N, self.K])
        for v in range(self.V):
            A = self.__decomp_metric(self.Metric[v][t-1])
            x_met = self.X[v] @ A
            y_met = self.Y[v][t] @ A
            dis_xy = dis_xy + dist.cdist(x_met, y_met)
        winner = np.argmin(dis_xy, axis=1)
        return winner

    # メトリック推定
    def __calc_metric(self, t, v, v_teach):
        # 誤差の共分散行列算出
        error = self.X[v] - self.Y[v][t, v_teach]
        e = (error.T @ error) / self.N
        self.Metric[v][t] = np.linalg.pinv(e)

    @staticmethod
    def __decomp_metric(s):
        # 行列のべき乗計算(固有値分解)
        la, u = np.linalg.eig(s)
        d = np.diag(la)

        # 1/2乗の算出
        a = u @ np.power(d, 0.5) @ u.T
        return a
