class Metrics(object):
    def __init__(self, metrics_file='dqn_metrics.txt'):
        self._episodic_t = 0
        self._rewards_accum = 0.0
        self._values_accum = 0.0
        self._metrics_file = metrics_file

    def write_metrics(self):
        metric = self.build_string()
        with open(self._metrics_file, 'w') as f:
            f.write('{}\n'.format(metric))

    def build_string(self):
        return '{}, {}'.format(self._rewards_accum,
                               self._values_accum)

    def inc_episodic_t(self):
        self._episodic_t += 1

    def add_to_rewards_accum(self, val):
        self._rewards_accum += val

    def add_to_values_accum(self, val):
        self._values_accum += val

    def get_episodic_t(self):
        return self._episodic_t

    def get_rewards_accum(self):
        return self._rewards_accum

    def get_values_accum(self):
        return self._values_accum

    def set_episodic_t(self, t):
        self._episodic_t = t

    def set_rewards_accum(self, rewards_accum):
        self._rewards_accum = rewards_accum

    def set_values_accum(self, values_accum):
        self._values_accum = values_accum
