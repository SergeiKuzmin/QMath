import numpy as np
from qmath import Propagator
from qmath import Pulse, Sine
from qmath import Fidelity, Gradient


Hd = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=complex)
Hx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Hy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
Hc = [Hx]

U_target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
fidelity = Fidelity(U_target)

T = 5.0
n_times = 200
timespan = np.linspace(0, T, n_times + 1)
print(timespan)
pulses = [Sine(1.0, 1.0, 5.0)]
control_pulse = Pulse(pulses)
propagator = Propagator(Hd, Hc)
list_u, der_u = propagator.evolution(control_pulse, timespan)
print([np.sum(np.dot(u, u.T.conj())) for u in list_u])
print([fidelity.fidelity(u) for u in list_u])
print(len(list_u))
print(der_u)
print(fidelity.infidelity(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)))
gradient = Gradient(U_target)
print(gradient.gradient(list_u[-1], der_u))
