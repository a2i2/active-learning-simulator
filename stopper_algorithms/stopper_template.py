from stopper_algorithms.stopper import Stopper


class NewStopper(Stopper):
    """
    Methods:

    - initialise: called during the initial sampling of the active learner
    - stopping_criteria: determines if a particular criteria for stopping has been met from the newly selected sample, called each iteration of the AL training
    - reset: resets any parameters and variables

    Attributes:

    - stop: specifics status of stopping criteria: 1 -> stop active learning, 0 -> continue active learning, -1 -> commence random learning
    - N: number of total documents in the dataset
    - confidence: confidence level parameter
    - verbose: specifics level of verbosity, True or False
    """
    def __init__(self, N, confidence, threshold=20, verbose=False):
        super().__init__(N, confidence, verbose)
        self.confidence = confidence
        self.verbose = verbose
        self.count = [0]
        self.threshold = int(threshold)

        # verbosity
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        """
        Updates the stopper for the initial samples: called each iteration of initial sampling.

        :param sample:
        :return:
        """
        self.stopping_criteria(sample)
        pass

    def stopping_criteria(self, sample):
        """
        Returns if the active learning should be terminated, i.e. a stopping criterion has been reached or whether
        random learning should begin instead.

        :param sample: newly selected sample to evaluate
        :return: 1 if the AL training should be stopped early, 0 if AL training should continue, -1 if random training should begin
        """
        current_count = self.count[-1] + sum(sample['y'])
        self.count.append(current_count)
        self.stop = current_count >= self.threshold
        self.out()
        return

    def get_eval_metrics(self):
        """
        Provides metrics specific to the stopper for later visualisation (optional).

        :return: list of metrics. Format for metric:
            metrics = [{'name': plot_name, 'x': (x_label, x_values), 'y': (y_label, y_values)}, ...]
        """
        metric = [{'name': 'stopper', 'x': ('iterations', list(range(len(self.count)))), 'y': ('relevant counts', self.count)},
                  {'name': 'stopper metric 2', 'x': ('iterations', list(range(len(self.count)))), 'y': ('relevant counts', self.count)}]
        return metric

    def verbose_output(self):
        print("\r" + "Hello: count={count}".format(count=self.count[-1]), end='')
        return
