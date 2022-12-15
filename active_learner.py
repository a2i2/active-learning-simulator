import numpy as np
from tqdm import tqdm


class ActiveLearner:
    def __init__(self, model, selector, stopper, batch_size=10, max_iter=100, evaluator=None, verbose=True):
        self.relevant_mask = None
        self.indice_mask = None
        self.N = None
        self.data = None
        self.data_indices = None
        self.model = model
        self.selector = selector
        self.stopper = stopper

        self.max_iter = max_iter
        self.batch_size = batch_size

        # handling progress and evaluator output
        if verbose:
            self.pbar = tqdm(total=self.max_iter, position=0, leave=True)

            def progress(active_learner):
                active_learner.pbar.update()
                print('Recall:', active_learner.evaluator.recall[-1])

            self.progress = progress

            def end_progress(active_learner):
                active_learner.pbar.close()

            self.end_progress = end_progress
        else:
            self.pbar = None
            self.progress = lambda *a: None
            self.end_progress = lambda *a: None
        if evaluator:
            self.evaluator = evaluator

            def initialise_evaluator(sample, test_data):
                evaluator.initialise(sample, test_data)

            self.initialise_evaluator = initialise_evaluator

            def update_evaluator(m, sample, test_data):
                evaluator.update(m, sample, test_data)

            self.update_evaluator = update_evaluator

            def reset_evaluator():
                evaluator.reset()

            self.reset_evaluator = reset_evaluator
        else:
            self.initialise_evaluator = lambda *a: None
            self.update_evaluator = lambda *a: None
            self.reset_evaluator = lambda *a: None

    # train (and test) active learner
    def train(self, data):
        """
        Training handler for the active learner

        :param data: training dataset DataFrame
        :return:
        """
        self.reset()
        self.initialise(data)
        self.initial_sampling()
        self.active_learn()
        return self.indice_mask, self.relevant_mask

    # initialise active learner parameters
    def initialise(self, data):
        """
        Initialise active learner parameters

        :param data: full dataset DataFrame
        """
        self.data = data
        self.N = len(data)
        # use new dataset indices instead of handling the data directly
        self.data_indices = np.arange(self.N)
        # create masks for training and testing instances
        self.indice_mask = np.zeros(self.N, dtype=np.uint8)
        self.relevant_mask = np.zeros(self.N, dtype=np.uint8)

    # initial sampling from test set to be training instances
    def initial_sampling(self):
        """
        Handles the initial selection / sampling of training instances. Keeps sampling until training sample contains
        instances belonging to every class
        """
        test_data = self.data
        test_indices = self.data_indices
        while True:
            # initial sampling
            sample_indices = self.selector.initial_select(test_data, test_indices)
            sample = self.data.iloc[sample_indices]

            # update mask to include initial training instances
            self.indice_mask[sample_indices] = 1
            self.relevant_mask[sample_indices] = sample['y']
            # get indices for training and testing instances
            test_indices = self.data_indices[self.indice_mask == 0]
            train_indices = self.data_indices[self.indice_mask == 1]
            # new test dataset excludes screened instances
            test_data = self.data.iloc[test_indices]

            # initialise, update stopper
            self.stopper.initialise(sample)
            # update evaluator
            self.initialise_evaluator(sample, self.data)

            # check if sample has two classes
            sample_sum = sum(sample['y'])
            if sample_sum != len(sample['y']) and sample_sum != 0:
                break

    # active learning loop
    def active_learn(self):
        """
        Handles the active learning loop: sample selection, model training, stopping
        """
        train_indices = []
        for i in range(self.max_iter):
            # get indices for training and testing instances
            test_indices = self.data_indices[self.indice_mask == 0]
            train_indices = self.data_indices[self.indice_mask == 1]
            if len(test_indices) == 0:
                break

            # add screened instances to training data
            train_data = self.data.iloc[train_indices]
            # new test dataset excludes screened instances
            test_data = self.data.iloc[test_indices]

            # train and test model
            self.model.train(train_data)
            preds = self.model.test(test_data)  # -model.test(test_data)[:, 1] ??

            # screen test instances
            sample_indices = self.selector.select(test_indices, preds)
            sample = self.data.iloc[sample_indices]

            # add screened instances to training set
            self.indice_mask[sample_indices] = 1
            self.relevant_mask[sample_indices] = sample['y']

            # update eval
            self.update_evaluator(self.model, sample, self.data)
            # print progress
            self.progress(self)

            # stopping criteria
            if self.stopper.stopping_criteria(sample):
                break

        # final model
        train_data = self.data.iloc[train_indices]
        self.model.train(train_data)

        self.end_progress(self)
        return

    # TODO random learning
    # random learning loop
    def random_learn(self):
        """
        Handles the active learning loop: sample selection, model training, stopping
        """
        train_indices = []
        for i in range(self.max_iter):
            # get indices for training and testing instances
            test_indices = self.data_indices[self.indice_mask == 0]
            train_indices = self.data_indices[self.indice_mask == 1]
            if len(test_indices) == 0:
                break

            # add screened instances to training data
            train_data = self.data.iloc[train_indices]
            # new test dataset excludes screened instances
            test_data = self.data.iloc[test_indices]

            # screen test instances
            sample_indices = self.selector.initial_select(test_data, test_indices)
            sample = self.data.iloc[sample_indices]

            # add screened instances to training set
            self.indice_mask[sample_indices] = 1
            self.relevant_mask[sample_indices] = sample['y']

            # update eval
            self.update_evaluator(self.model, sample, self.data)
            # print progress
            self.progress(self)

            # stopping criteria
            if self.stopper.stopping_criteria(sample):
                break

        # final
        train_data = self.data.iloc[train_indices]
        self.end_progress(self)
        return

    def reset(self):
        """
        Reset active learner and enclosed structures
        """
        self.model.reset()
        self.selector.reset()
        self.stopper.reset()
        self.reset_evaluator()
        return
