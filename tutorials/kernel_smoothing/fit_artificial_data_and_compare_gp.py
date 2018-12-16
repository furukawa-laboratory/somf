import numpy as np
import matplotlib.pyplot as plt

from libs.models import KernelSmoothing


def _main():
    # Set random seed
    seed = 10
    np.random.seed(seed)

    # Generate artificial dataset
    nb_samples = 100
    x_noise_sigma = 0.1
    x = np.random.rand(nb_samples) * 2.0 - 1.0
    y = func(x) + np.random.normal(0.0, x_noise_sigma, x.shape)

    # Set parameter of ks
    sigma = 0.1

    # Set parameter to visualize
    nb_new_samples = 200
    xnew = np.linspace(-1.2,1.2,nb_new_samples)
    f_true = func(xnew)

    # Fit and predict by kernel smoothing
    ks = KernelSmoothing(sigma=sigma)
    ks.fit(x,y)
    f_ks = ks.predict(xnew)

    # Plot training samples and estimated function
    fig = plt.figure(figsize=[5,3])
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y,label='samples', s=8)
    ax.plot(xnew,f_ks,label='ks')
    ax.plot(xnew,f_true,label='true')
    plt.legend()
    plt.show()


def func(x):
    return 0.6 * x + 0.3 * (x**2.0) + 0.7 * (x**3.0)




if __name__ == '__main__':
    _main()