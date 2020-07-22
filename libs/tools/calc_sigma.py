import numpy as np
from enum import Enum, auto


class SigmaMode(Enum):
    LINEAR = auto()
    EXPONENTIAL = auto()


def calc_sigma(mode: SigmaMode, sigma_max: float, sigma_min: float, tau: float, epoch: int) -> float:
    if mode is SigmaMode.LINEAR:
        return __calc_sigma_linear(sigma_max, sigma_min, tau, epoch)
    elif mode is SigmaMode.EXPONENTIAL:
        return __calc_sigma_exp(sigma_max, sigma_min, tau, epoch)
    else:
        raise NotImplementedError()


def __calc_sigma_exp(sigma_max: float, sigma_min: float, tau: float, epoch: int) -> float:
    return sigma_min + (sigma_max - sigma_min) * np.exp(-epoch / tau)


def __calc_sigma_linear(sigma_max: float, sigma_min: float, tau: float, epoch: int) -> float:
    return max(sigma_min, sigma_min + (sigma_max - sigma_min) * (1 - (epoch / tau)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    sigma_max = 1.0
    sigma_min = 0.1
    tau = 10.0
    max_epoch = 100

    x = np.arange(max_epoch)

    for sigma_mode in SigmaMode:
        y = [calc_sigma(sigma_mode, sigma_max, sigma_min, tau, epoch) for epoch in x]
        ax.plot(x, y, label=sigma_mode.name)

    ax.legend()
    plt.show()
