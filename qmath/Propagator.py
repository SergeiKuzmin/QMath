import numpy as np
from qmath.Stepper import Stepper
import copy


class Propagator(object):
    def __init__(self, hd, hc):
        self.hd = hd
        self.hc = hc
        self.stepper = Stepper(self.hd, self.hc)
        self.u = np.eye(hd.shape[0], dtype=complex)
        self.der_u = None
        self.list_u = None

    def evolution(self, pulse, timespan):
        list_u = []
        u = np.eye(self.hd.shape[0], dtype=complex)
        der_u = []
        for k in range(len(pulse.der_pulse_time(timespan[-1]))):
            der_u.append(np.zeros_like(self.hd, dtype=complex))
        for t in range(len(timespan[:-1])):
            dt = timespan[t + 1] - timespan[t]
            u, der_u = self.stepper.do_step(u, der_u, pulse, timespan[t], dt)
            list_u.append(u)
        self.u = u
        self.der_u = der_u
        self.list_u = list_u
        print('INFO: full_evolution')
        return list_u, der_u
