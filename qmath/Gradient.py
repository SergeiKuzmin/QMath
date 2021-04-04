import numpy as np


class Gradient(object):
    def __init__(self, u_target):
        self.u_target = u_target

    def gradient(self, u, der_u):
        grad = []
        for der in der_u:
            z = np.trace(np.dot(self.u_target.T.conj(), u))
            y = np.trace(np.dot(self.u_target.T.conj(), der))
            grad.append((-2 / (u.shape[0] ** 2)) * (z.conj() * y).real)
        return np.array(grad)
