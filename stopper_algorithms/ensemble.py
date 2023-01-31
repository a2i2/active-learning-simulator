from stopper_algorithms.stopper import Stopper


class Ensembler(Stopper):
    def __init__(self, N, tau_target=0.95, *stoppers, verbose=False):
        super().__init__(N, tau_target, verbose)
        self.stoppers = []
        for stopper in stoppers:
            self.stoppers.append(globals()[stopper](N, tau_target))
        # verbosity
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        for stopper in self.stoppers:
            stopper.initialise(sample)
        return

    def stopping_criteria(self, sample):
        self.stop = False
        for stopper in self.stoppers:
            self.stop *= stopper.stopping_criteria(sample)
        return self.stop

    def get_eval_metrics(self):
        metrics = []
        for stopper in self.stoppers:
            metric = stopper.get_eval_metrics()
            if metric:
                metrics.append(*metric)
        return metrics

    def verbose_output(self):
        for stopper in self.stoppers:
            stopper.verbose_output()
        return
