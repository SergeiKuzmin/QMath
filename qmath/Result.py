class Result(object):
    def __init__(self, timespan):
        self.timespan = timespan
        self.pulse_iterations = []
        self.infidelity_iterations = []

    def add_iteration(self, pulse_time, infidelity):
        self.pulse_iterations.append(pulse_time)
        self.infidelity_iterations.append(infidelity)

    def get_result(self):
        return self.pulse_iterations
