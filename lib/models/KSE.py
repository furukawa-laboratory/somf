from lib.models.flab.KSE0331 import KSE as KSE0331


def KSE(version, X):
    if version == "0331":
        return KSE0331(X)
    else:
        raise ValueError("Undefined Version {}".format(version))
