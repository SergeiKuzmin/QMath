import numpy as np
from qmath import Pulse, Sine, Fidelity, Gradient, GOAT, Propagator, Stepper

Hd = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=complex)
Hx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Hy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
Hz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
Hc = [Hx, Hy, Hz]

T = 4.0
n_times = 2000
timespan = np.linspace(0, T, n_times + 1)

U_target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
fidelity = Fidelity(U_target)
gradient = Gradient(U_target)
propagator = Propagator(Hd, Hc)
pulse = Pulse([Sine(0.1, 0.1, 5.0), Sine(0.1, 0.1, 5.0), Sine(0.1, 0.1, 5.0)])
goat = GOAT(Hd, Hc, pulse, propagator, fidelity, gradient, timespan)
goat.optimizer(100)
print(fidelity.infidelity(propagator.u))
