import numpy as np
from qmath import Propagator
from qmath import Pulse, Sine


Hd = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=complex)
Hx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Hc = [Hx]

T = 5.0
n_times = 200
timespan = np.linspace(0, T, n_times + 1)
pulses = [Sine(1.0, 1.0, 1.0)]
control_pulse = Pulse(pulses)
propagator = Propagator(Hd, Hc)
list_u, der_u = propagator.evolution(control_pulse, timespan)
print(list_u)
print(len(list_u))
print(len(der_u))
