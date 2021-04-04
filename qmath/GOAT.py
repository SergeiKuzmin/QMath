import numpy as np
from scipy.optimize import minimize
from qmath import Fidelity, Gradient, Propagator


def func(params, pulse, propagator, fidelity, gradient, timespan):
    pulse.set_params(params)
    propagator.evolution(pulse, timespan)
    fid = fidelity.infidelity(propagator.u)
    print('fid = ', fid)
    return fid


def func_grad(params, pulse, propagator, fidelity, gradient, timespan):
    pulse.set_params(params)
    grad = gradient.gradient(propagator.u, propagator.der_u)
    print('grad = ', grad)
    return grad


class GOAT(object):
    def __init__(self, hd, hc, pulse, propagator, fidelity, gradient, timespan):
        self.hd = hd
        self.hc = hc
        self.pulse = pulse
        self.propagator = propagator
        self.fidelity = fidelity
        self.gradient = gradient
        self.timespan = timespan

    def optimizer(self, num_of_iter):
        x0 = self.pulse.get_params()
        for iterations in range(num_of_iter):
            res = minimize(func, x0, args=(self.pulse, self.propagator, self.fidelity,
                                           self.gradient, self.timespan), method='BFGS',
                           options={'disp': False, 'maxiter': 1})
            print(res.x)
            x0 = res.x
            # print(self.fidelity.infidelity(self.propagator.u))
