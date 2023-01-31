from stopper_algorithms.stopper import Stopper


class SampleProportion(Stopper):
    """
    Determines an estimate proportion of relevant documents in the dataset. Calculates the number of relevants documents
    required to be screened; AL stops when (most) relevant documents are thought to be screened.

    Attributes:

    - tau_target: target recall
    - r_total: total number of relevant documents in the dataset
    - r_AL: current number of relevant documents screened by the active learner
    - N: total number of documents in the dataset
    - verbose: when true, outputs progress
    """
    def __init__(self, N, tau_target=0.95, verbose=True):
        super().__init__(N, tau_target, verbose)
        self.n_sample = 0
        self.r_sample = 0
        self.tau_target = tau_target
        self.r_total = 0
        self.r_AL = 0
        self.k = 0
        self.N = N
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        self.n_sample += len(sample)
        self.r_sample += sum(sample['y'])
        # baseline inclusion rate
        BIR = self.r_sample / self.n_sample
        # estimate total number of relevants
        self.r_total = BIR * self.N
        self.stopping_criteria(sample)
        return

    def stopping_criteria(self, sample):
        self.k = sum(sample['y'])
        self.r_AL = self.r_AL + self.k
        if self.r_total > 0:
            self.stop = (self.r_AL >= self.tau_target * self.r_total)
        self.out()
        return

    def verbose_output(self):
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return
