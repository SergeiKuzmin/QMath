import numpy as np


class Stepper(object):
    def __init__(self, hd, hc):
        self.hd = hd
        self.hc = hc
        self.sys_cur = None
        self.sys_prev = None
        self.x = None
        self.c = None

    def set_system_cur(self, controls, pulse, t):
        self.c = len(pulse.der_pulse_time(t))
        ham = self.hd
        for k, control in enumerate(controls):
            ham += control * self.hc[k]
        der_ham = []
        index = 0
        for k, pulse in enumerate(pulse.pulses):
            for param in range(pulse.number_of_params):
                der_ham.append(pulse.der_pulse_time(t)[index] * self.hc[k])
                index += 1
        self.sys_cur = np.kron(np.eye(self.c + 1, dtype=complex), ham)
        for k in range(self.c):
            matrix = np.zeros_like(np.eye(self.c + 1, dtype=complex), dtype=complex)
            matrix[0][k + 1] = 1.0
            self.sys_cur += np.kron(matrix, der_ham[k])
        self.sys_cur = -1j * self.sys_cur

    def set_system_prev(self, controls, pulse, t):
        self.c = len(pulse.der_pulse_time(t))
        ham = self.hd
        for k, control in enumerate(controls):
            ham += control * self.hc[k]
        der_ham = []
        index = 0
        for k, pulse in enumerate(pulse.pulses):
            for param in range(pulse.number_of_params):
                der_ham.append(pulse.der_pulse_time(t)[index] * self.hc[k])
                index += 1
        self.sys_prev = np.kron(np.eye(self.c + 1, dtype=complex), ham)
        for k in range(self.c):
            matrix = np.zeros_like(np.eye(self.c + 1, dtype=complex), dtype=complex)
            matrix[0][k + 1] = 1.0
            self.sys_prev += np.kron(matrix, der_ham[k])
        self.sys_prev = -1j * self.sys_prev

    def set_x(self, u, der_u):
        vector = np.zeros(self.c + 1, dtype=complex)
        vector[0] = 1.0
        self.x = np.kron(vector, u)
        for k in range(self.c):
            vector = np.zeros(self.c + 1, dtype=complex)
            vector[k + 1] = 1.0
            self.x += np.kron(vector, der_u[k])

    def do_step(self, u, der_u, pulse, t, dt):
        print('do_step')
        self.set_system_prev(pulse.pulse_time(t), pulse, t)
        self.set_system_cur(pulse.pulse_time(t + dt), pulse, t + dt)
        self.set_x(u, der_u)
        print(self.sys_cur.shape)
        print(self.x.shape)
        x = np.dot(np.linalg.inv(np.eye(self.sys_cur.shape[0]) - 0.5 * dt * self.sys_cur),
                   np.dot(np.eye(self.sys_prev.shape[0]) + 0.5 * dt * self.sys_prev, self.x.T))
        u_new = x[0:u.shape[0]]
        der_u_new = []
        for k, der in enumerate(der_u):
            der_u_new.append(x[u.shape[0] * (k + 1):u.shape[0] * (k + 1) + u.shape[0]])
        return u_new, der_u_new
