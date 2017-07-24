from lib.models.flab.KSE0331 import KSE as KSE0331
from lib.models.flab.KSEstandard import KSE as KSEstandard
from lib.models.flab.KSE0428 import KSE as KSE0428
from lib.models.flab.KSE0524 import KSE as KSE0524


def KSE(version, X, latent_dim, init):
    if version == "0331":
        return KSE0331(X, latent_dim, init)
    elif version == "standard":
        return KSEstandard(X, latent_dim, init)
    elif version == "0428":
        return KSE0428(X, latent_dim, init)
    elif version == "0524":
        return KSE0524(X, latent_dim, init)
    else:
        raise ValueError("Undefined Version {}".format(version))
