import numpy as np


class Pulse(object):
    def __init__(self, pulses):
        self.pulses = pulses

    def get_pulses(self):
        return self.pulses

    def get_number_params(self):
        number_params = 0
        for pulse in self.pulses:
            number_params += pulse.number_of_params
        return number_params

    def get_params(self):
        params = []
        for pulse in self.pulses:
            params += pulse.get_params()
        return np.array(params)

    def set_params(self, list_of_params):
        index = 0
        for pulse in self.pulses:
            pulse.set_params(*list_of_params[index:index + pulse.number_of_params])
            index += pulse.number_of_params

    def pulse_time(self, timespan):
        controls = []
        for pulse in self.pulses:
            controls.append(pulse.pulse_time(timespan))
        # print('pulse_time = ', np.array(controls))
        return np.array(controls)

    def der_pulse_time(self, timespan):
        der_controls = []
        for pulse in self.pulses:
            der_controls += pulse.der_pulse_time(timespan)
        # print('der_pulse_time = ', np.array(der_controls))
        return np.array(der_controls)


class Sine(object):
    def __init__(self, a, b, omega):
        self.a = a
        self.b = b
        self.omega = omega
        self.number_of_params = 3

    def get_params(self):
        return [self.a, self.b, self.omega]

    def set_params(self, a, b, omega):
        self.a = a
        self.b = b
        self.omega = omega

    def pulse_time(self, timespan):
        return self.a * np.cos(self.omega * timespan) + self.b * np.sin(self.omega * timespan)

    def der_pulse_time(self, timespan):
        der_a = np.cos(self.omega * timespan)
        der_b = np.sin(self.omega * timespan)
        der_omega = timespan * (-self.a * np.sin(self.omega * timespan) +
                                self.b * np.cos(self.omega * timespan))
        return [der_a, der_b, der_omega]
