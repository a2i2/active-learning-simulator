import matplotlib.pyplot as plt
import numpy as np


class Evaluator:
    def __init__(self, data, verbose=True):
        self.n = [0]  # number of documents sampled during training
        self.r_AL = [0]  # number of relevants seen
        self.r_total = sum(data['y'])  # total number of actual relevants in dataset
        self.recall = [0]  # actual recall during training
        self.N = len(data)  # total number of documents in dataset
        self.N_AL = [0]  # total number of documents seen during training
        self.k = [0]  # number of relevants sampled during training
        self.work_save = [0.0]  # actual work save during training
        self.tau_model = [0.0]  # recall of the ML model during training
        if verbose:
            self.out = self.output_results
        else:
            self.out = lambda *a: None

    def initialise(self, sample, test_data):
        self.n.append(len(sample['y']))
        self.k.append(sum(sample['y']))
        self.r_AL.append(self.r_AL[-1] + self.k[-1])
        self.recall.append(self.r_AL[-1] / self.r_total)
        self.N_AL.append(self.N_AL[-1] + self.n[-1])
        self.work_save.append(1 - self.N_AL[-1] / self.N)
        self.tau_model.append(0)
        return

    def update(self, model, sample, test_data):
        """
        Update evaluator statistics

        :param model: machine learning model used in active learning
        :param sample: new sample instances
        :param test_data: remaining testing data
        """
        self.n.append(len(sample['y']))
        self.k.append(sum(sample['y']))
        self.r_AL.append(self.r_AL[-1] + self.k[-1])
        self.recall.append(self.r_AL[-1] / self.r_total)
        self.N_AL.append(self.N_AL[-1] + self.n[-1])
        self.work_save.append(1 - self.N_AL[-1] / self.N)
        preds = model.predict(test_data)
        self.tau_model.append(sum(test_data['y'] * preds) / self.r_total)
        return

    def reset(self):
        """
        Reset evaluator statistics and parameters
        """
        self.n = [0]
        self.r_AL = [0]
        self.recall = [0]
        self.N_AL = [0]
        self.k = [0]
        self.work_save = [0]
        self.tau_model = [0]
        return

    def output_results(self, model, test_data):
        print('\nRecall:', self.recall[-1])
        print('Work save:', self.work_save[-1])
        print('Relevants found:', self.r_AL[-1])

        y = test_data['y']
        print('Actual number of relevants:', sum(y))
        print('Total reviews screened:', self.N_AL[-1])
        print('Total reviews:', self.N)

        preds = model.predict(test_data)
        print('Model predicted relevants:', sum(preds))
        print('\n')

    def get_eval_metrics(self):
        return [{'name': 'recall', 'x': ('documents seen', self.N_AL), 'y': ('recall', self.recall)},
                {'name': 'model recall', 'x': ('documents seen', self.N_AL[len(self.N_AL) - len(self.tau_model):]), 'y': ('model recall', self.tau_model)}]


# TODO selector, stopper should have their own output: append to a results list
def visualise_training(results):
    """
    Visualises the performance of the system on a dataset

    :param results: {name : name, x : (name, vals), y : (name, vals)}
    """
    for i, result in enumerate(results):
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(result['x'][0])
        ax1.set_ylabel(result['y'][0])
        ax1.plot(result['x'][1], result['y'][1], label=result['y'][0])
        ax1.tick_params(axis='y')
        fig1.tight_layout()
        ax1.set_title(result['name'])
        legend = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        ax1.figure.savefig('output_{0}.png'.format(i), dpi=300)


def visualise_results(evaluators):
    """
    Visualise the results across several datasets

    :param evaluators: list of evaluators, one for each dataset training
    """
    recalls = np.zeros(shape=(len(evaluators), 1))
    work_saves = np.zeros(shape=(len(evaluators), 1))
    colours = np.zeros(shape=(len(evaluators), 1))
    N_min = evaluators[0].N
    N_max = evaluators[0].N

    for i, evaluator in enumerate(evaluators):
        recalls[i] = evaluator.recall[-1]
        work_saves[i] = evaluator.work_save[-1]
        colours[i, :] = [evaluator.N]
        N_min = min(evaluator.N, N_min)
        N_max = min(evaluator.N, N_max)

    # normalise colours
    colours = (colours - N_min) / N_max * 255.0

    fig = plt.figure(constrained_layout=True)

    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)
    #ax.set_title('Recall - work save')
    ax.set_xlabel('Work save')
    ax.set_ylabel('Recall')
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # p = ax.scatter(work_saves, recalls, c=colours, alpha=0.5)
    p = scatter_hist(work_saves, recalls, ax, colours, ax_histx, ax_histy)

    fig.colorbar(p, ax=ax)

    plt.show()
    ax.figure.savefig('recall-work.png', dpi=300)


def scatter_hist(x, y, ax, colours, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    p = ax.scatter(x, y, c=colours, alpha=0.5)

    # now determine nice limits by hand:
    binwidth = 0.01
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)

    plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(x.mean() * 1.1, max_ylim * 0.1, 'Mean: {:.2f}'.format(x.mean()))

    ax_histy.hist(y, bins=bins, orientation='horizontal')

    plt.axhline(y.mean(), color='k', linestyle='dashed', linewidth=1)
    min_xlim, max_xlim = plt.xlim()
    plt.text(max_xlim * 0.7, y.mean() * 0.95, 'Mean: {:.2f}'.format(y.mean()))

    return p
