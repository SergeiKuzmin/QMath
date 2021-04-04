import numpy as np


class Fidelity(object):
    def __init__(self, u_target):
        self.u_target = u_target

    def fidelity(self, u):
        return np.abs(np.trace(np.dot(self.u_target.T.conj(), u))) ** 2 / u.shape[0] ** 2

    def infidelity(self, u):
        return 1 - self.fidelity(u)
