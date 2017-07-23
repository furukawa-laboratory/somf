from lib.models.flab.KSE0331 import KSE as KSE0331
from lib.models.flab.KSEstandard import KSE as KSEstandard


def KSE(version, X, latent_dim, init):
    if version == "0331":
        return KSE0331(X, latent_dim, init)
    elif version == "standard":
        return KSEstandard(X, latent_dim, init)
    else:
        raise ValueError("Undefined Version {}".format(version))
