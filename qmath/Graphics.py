import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch


class Plot(object):
    def __init__(self, result, propagator):
        self.result = result
        self.propagator = propagator

    def show_evolution_pulse(self, number_of_control):
        list_controls = self.result.get_result()
        fig, ax = plt.subplots()
        for i, controls in enumerate(list_controls):
            plt.plot(self.result.timespan, controls[number_of_control], color='blue', lw=2,
                     alpha=0.2 + 0.3 * (i / len(list_controls)))
        plt.tick_params(which='major', direction='in')
        plt.tick_params(which='minor', direction='in')
        ax.grid()
        ax.minorticks_off()
        plt.xlabel('time', fontsize=15)
        plt.ylabel('pulse: '+str(number_of_control), fontsize=15)
        plt.plot()
        plt.show()

    @staticmethod
    def xyz(state):
        state = state.reshape((1, 2))
        rho = np.dot(state.T.conj(), state)
        x = 2 * np.real(rho[0][1])
        y = 2 * np.imag(rho[1][0])
        z = np.real(rho[0][0] - rho[1][1])
        return [x, y, z]

    def bloch(self):
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))
        bloch = Bloch(fig=fig, axes=ax)
        # state = |0>
        state_0 = np.array([1.0, 0.0], dtype=complex)
        point_x = []
        point_y = []
        point_z = []
        for u in self.propagator.list_u:
            xyz = self.xyz(np.dot(u, state_0))
            point_x.append(xyz[0])
            point_y.append(xyz[1])
            point_z.append(xyz[2])
        bloch.add_points([point_x, point_y, point_z], 'l')
        bloch.render(fig=fig, axes=ax)
        plt.show()

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))
        bloch = Bloch(fig=fig, axes=ax)
        # state = |1>
        state_0 = np.array([0.0, 1.0], dtype=complex)
        point_x = []
        point_y = []
        point_z = []
        for u in self.propagator.list_u:
            xyz = self.xyz(np.dot(u, state_0))
            point_x.append(xyz[0])
            point_y.append(xyz[1])
            point_z.append(xyz[2])
        bloch.add_points([point_x, point_y, point_z], 'l')
        bloch.render(fig=fig, axes=ax)
        plt.show()

    def show_iterations(self):
        fig, ax = plt.subplots()
        plt.plot(self.result.infidelity_iterations, color='blue', lw=2)
        plt.tick_params(which='major', direction='in')
        plt.tick_params(which='minor', direction='in')
        ax.grid()
        ax.minorticks_off()
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Infidelity', fontsize=15)
        plt.yscale('log')
        plt.plot()
        plt.show()
