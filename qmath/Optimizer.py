import numpy as np
from scipy.optimize import minimize
from qmath import Fidelity, Gradient, Propagator


def func(params, pulse, propagator, fidelity, gradient, timespan):
    pulse.set_params(params)
    propagator.evolution(pulse, timespan)
    fid = fidelity.infidelity(propagator.u)
    return fid


def func_grad(params, pulse, propagator, fidelity, gradient, timespan):
    pulse.set_params(params)
    grad = gradient.gradient(propagator.u, propagator.der_u)
    return grad


class Optimizer(object):
    def __init__(self, hd, hc, pulse, propagator, fidelity, gradient, timespan, result):
        self.hd = hd
        self.hc = hc
        self.pulse = pulse
        self.propagator = propagator
        self.fidelity = fidelity
        self.gradient = gradient
        self.timespan = timespan
        self.result = result

    def optimizer(self, num_of_iter):
        x0 = self.pulse.get_params()
        for iterations in range(num_of_iter):
            self.result.add_iteration(self.pulse.pulse_time(self.timespan),
                                      self.fidelity.infidelity(self.propagator.u))
            res = minimize(func, x0, args=(self.pulse, self.propagator, self.fidelity,
                                           self.gradient, self.timespan), method='BFGS',
                           jac=func_grad, options={'disp': False, 'maxiter': 1})
            x0 = res.x
            self.pulse.set_params(x0)
            print('Iterations: ', iterations)
            print('Infidelity: ', self.fidelity.infidelity(self.propagator.u))
