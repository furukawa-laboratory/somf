
class KSE(object):
    def __init__(self, X):
        self.X = X.copy()
        print("hello")

    def fit(self):
        print("fit")
        print(self.X)
