import numpy as np

def create_zeta(zeta_min, zeta_max, latent_dim, resolution, include_min_max=True, return_step=False):
    mesh1d, step = np.linspace(zeta_min, zeta_max, resolution, endpoint=include_min_max, retstep=True)
    if include_min_max:
        pass
    else:
        mesh1d += step / 2.0

    if latent_dim == 1:
        Zeta = mesh1d
    elif latent_dim == 2:
        Zeta = np.meshgrid(mesh1d, mesh1d)
    else:
        raise ValueError("invalid latent dim {}".format(latent_dim))

    Zeta = np.dstack(Zeta).reshape(-1, latent_dim)
    if return_step:
        return Zeta, step
    else:
        return Zeta
