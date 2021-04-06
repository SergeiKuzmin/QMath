import numpy as np
from qmath import Pulse, Sine, Fidelity, Gradient, Optimizer, Propagator, Stepper, Result, Plot

Hd = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=complex)
Hx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Hy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
Hz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
Hc = [Hx, Hy, Hz]

T = 4.0
n_times = 200
timespan = np.linspace(0, T, n_times + 1)

U_target = (1 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
fidelity = Fidelity(U_target)
gradient = Gradient(U_target)
propagator = Propagator(Hd, Hc)
pulse = Pulse([Sine(0.1, 0.1, 4.0), Sine(0.1, 0.1, 4.0), Sine(0.1, 0.1, 4.0)])
result = Result(timespan)
plot = Plot(result, propagator)
optimizer = Optimizer(Hd, Hc, pulse, propagator, fidelity, gradient, timespan, result)
optimizer.optimizer(10)
print(fidelity.infidelity(propagator.u))
plot.show_evolution_pulse(0)
plot.bloch()
plot.show_iterations()
