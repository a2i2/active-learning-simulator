from stopper_algorithms.stopper import Stopper


class ConsecutiveCount(Stopper):
    """
    Ceases active learning when consecutive number of irrelevant documents are seen

    Attributes:

    - N: total number of documents in the full dataset
    - threshold: maximum number of allowable irrelevant documents
    - count: number of current irrelevant documents seen consecutively
    - verbose: when true, outputs progress
    """
    def __init__(self, N, tau_target=0.95, threshold=0.05, verbose=False):
        super().__init__(N, tau_target, verbose)
        self.N = N
        self.threshold = float(threshold) * self.N
        self.count = 0
        self.counts = []
        self.k = 0
        self.r_AL = 0
        if verbose:
            self.out = self.verbose_output
        else:
            self.out = lambda *a: None

    def initialise(self, sample):
        self.count = 0
        return

    def stopping_criteria(self, sample):
        self.k = sum(sample['y'])
        self.r_AL += self.k
        self.stop = False
        # calculate p-value for every addition of sample instances
        for i in range(len(sample)):
            y = sample.iloc[i]['y']
            self.count = (not y) * (self.count + 1)
            self.counts.append(self.count)
            if self.count >= self.threshold:
                self.stop = True
                break
        return

    def get_eval_metrics(self):
        return [{'name': 'stopper', 'x': ('iterations', list(range(len(self.counts)))), 'y': ('irrelevant counts', self.counts)}]

    def verbose_output(self):
        print("\r" + 'Number of relevants seen in sample:', str(self.k), 'for a total of', str(self.r_AL), end='')
        return
