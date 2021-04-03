import numpy as np


class Fidelity(object):
    def __init__(self, u_target):
        self.u_target = u_target

    def fidelity(self, u):
        return np.trace(np.abs(np.dot(self.u_target.T.conj(), u)) ** 2 /
                        np.trace(np.dot(self.u_target.T.conj(), self.u_target)) *
                        np.trace(np.dot(u.T.conj(), u)))

    def infidelity(self, u):
        return 1 - self.fidelity(u)
