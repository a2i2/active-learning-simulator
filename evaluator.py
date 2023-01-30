import os
import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.express as px
import pandas as pd


class Evaluator:
    def __init__(self, data, verbose=True):
        self.N = len(data)  # total number of documents in dataset
        self.r_total = sum(data['y'])  # total number of actual relevants in dataset

        self.n = [0]  # number of documents sampled during training
        self.r_AL = [0]  # number of relevants seen
        self.recall = [0]  # actual recall during training
        self.N_AL = [0]  # total number of documents seen during training
        self.k = [0]  # number of relevants sampled during training

        self.work_save = [0.0]  # actual work save during training
        self.tau_model = [0.0]  # recall of the ML model during training
        self.screen_indices = []  # indices of the screened instances in order of screening

        if verbose:
            self.out = self.output_results
        else:
            self.out = lambda *a: None

    def initialise(self, sample, test_data):
        """
        Initialise evaluator object
        :param sample: initial dataset sample
        :param test_data: testing data (full dataset)
        """
        self.n.append(len(sample['y']))
        self.k.append(sum(sample['y']))
        self.r_AL.append(self.r_AL[-1] + self.k[-1])
        self.recall.append(self.r_AL[-1] / self.r_total)
        self.N_AL.append(self.N_AL[-1] + self.n[-1])
        self.work_save.append(1 - self.N_AL[-1] / self.N)
        self.tau_model.append(0)
        self.screen_indices += list(sample.index.values)
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
        preds = model.predict(test_data['x'].apply(pd.Series), test_data['y'])
        self.tau_model.append(sum(test_data['y'] * preds) / self.r_total)
        self.screen_indices += list(sample.index.values)
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
        self.screen_indices = []
        return

    def output_results(self, model, test_data):
        """
        Print results to console.
        :param model: trained Machine Learning model
        :param test_data: testing data (full dataset)
        :return:
        """
        preds = model.predict(test_data['x'].apply(pd.Series), test_data['y'])
        output_string = ''
        output_string += '\nRecall: {0}'.format(self.recall[-1])
        output_string += '\nWork save: {0}'.format(self.work_save[-1])
        output_string += '\nRelevants found: {0}'.format(self.r_AL[-1])
        output_string += '\nActual number of relevants: {0}'.format(sum(test_data['y']))
        output_string += '\nTotal reviews screened: {0}'.format(self.N_AL[-1])
        output_string += '\nTotal reviews: {0}'.format(self.N)
        output_string += '\nModel predicted relevants: {0}'.format(sum(preds))
        output_string += '\n'

        print(output_string)
        return output_string

    def get_eval_metrics(self):
        return [{'name': 'recall', 'x': ('documents seen', self.N_AL), 'y': ('recall', self.recall)},
                {'name': 'model recall', 'x': ('documents seen', self.N_AL[len(self.N_AL) - len(self.tau_model):]), 'y': ('model recall', self.tau_model)}]


def output_results(active_learners, output_path, output_metrics=None):
    if output_metrics is None:
        output_metrics = []
    overall = [{'name': 'work save - recall', 'x': ('work_save', []), 'y': ('recall', []), 'colours': []}]

    for i, AL in enumerate(active_learners):
        results = []
        evaluator = AL.evaluator
        model = AL.model
        selector = AL.selector
        stopper = AL.stopper

        results.append(
            {'name': 'documents_sampled', 'x': ('iterations', list(range(len(evaluator.n)))), 'y': ('documents', evaluator.n)})
        results.append(
            {'name': 'relevants_sampled', 'x': ('iterations', list(range(len(evaluator.k)))), 'y': ('documents', evaluator.k)})

        results.append(
            {'name': 'documents_seen', 'x': ('iterations', list(range(len(evaluator.N_AL)))), 'y': ('documents', evaluator.N_AL)})
        results.append(
            {'name': 'relevants_seen', 'x': ('iterations', list(range(len(evaluator.r_AL)))), 'y': ('documents', evaluator.r_AL)})

        results.append(
            {'name': 'true_recall', 'x': ('iterations', list(range(len(evaluator.recall)))), 'y': ('recall', evaluator.recall)})
        results.append(
            {'name': 'true_work_save', 'x': ('iterations', list(range(len(evaluator.work_save)))), 'y': ('work save', evaluator.work_save)})
        results.append(
            {'name': 'model_recall', 'x': ('iterations', list(range(len(evaluator.tau_model)))), 'y': ('recall', evaluator.tau_model)})

        results.append(
            {'name': 'screened_indices', 'x': ('iterations', list(range(len(evaluator.screen_indices)))), 'y': ('indices', list(map(int, evaluator.screen_indices)))})

        # model metrics
        model_metrics = model.get_eval_metrics()
        if model_metrics:
            for metric in model_metrics:
                metric['name'] = 'model'
                results.append(metric)

        # selector metrics
        selector_metrics = selector.get_eval_metrics()
        if selector_metrics:
            for metric in selector_metrics:
                metric['name'] = 'selector'
                results.append(metric)

        # stopper metrics
        stopper_metrics = stopper.get_eval_metrics()
        if stopper_metrics:
            for metric in stopper_metrics:
                metric['name'] = 'stopper'
                results.append(metric)

        # create dataset output path
        output_dataset_path = "{path}/dataset_{name}/".format(path=output_path, name=(i+1))
        if not os.path.isdir(output_dataset_path):
            os.makedirs(output_dataset_path)
        output_name = "{path}/{name}.json".format(path=output_dataset_path, name="results")
        with open(output_name, 'w') as f:
            json.dump(results, f)

        overall[0]['x'][1].append(evaluator.work_save[-1])
        overall[0]['y'][1].append(evaluator.recall[-1])
        overall[0]['colours'].append(AL.N)

        # plot desired metrics
        for result in results:
            if result['name'] in output_metrics:
                ax = metric_plot(result)
                ax.figure.savefig("{path}/{metric}.png".format(path=output_dataset_path, metric=result['name']), dpi=300)

    #
    recalls = overall[0]['y'][1]
    mean_recall = sum(recalls) / len(recalls)
    min_recall = min(recalls)

    work_saves = overall[0]['x'][1]
    mean_work_save = sum(work_saves) / len(work_saves)
    min_work_save = min(work_saves)

    overall.append({'mean_recall': mean_recall, 'min_recall': min_recall, 'mean_work_save': mean_work_save, 'min_work_save': min_work_save})

    output_name = "{path}/{name}.json".format(path=output_path, name="overall")
    with open(output_name, 'w') as f:
        json.dump(overall, f)

    ax = scatter_plot(overall[0], colour_label=overall[0]['colours'], marginal=True)
    ax.write_html("{path}/{fig_name}.html".format(path=output_path, fig_name="overall"))


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

    # normalise colours between 0-255
    colours = (colours - N_min) / N_max * 255.0

    # format figure
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)
    ax.set_xlabel('Work save')
    ax.set_ylabel('Recall')

    # create distribution axes
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # create distribution plots
    p = scatter_hist(work_saves, recalls, ax, colours, ax_histx, ax_histy)

    # create colour bar
    fig.colorbar(p, ax=ax)

    # show and save plot to file
    #plt.show()
    #ax.figure.savefig('recall-work.png', dpi=300)
    return ax


def visualise_configs(work_saves, recalls):
    N = len(work_saves)
    # normalise colours between 0-255
    colours = (np.arange(0, N)) / N * 255.0

    # format figure
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)
    ax.set_xlabel('Work save')
    ax.set_ylabel('Recall')

    # create distribution axes
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # create distribution plots
    p = scatter_hist(np.array(work_saves), np.array(recalls), ax, colours, ax_histx, ax_histy)

    # create colour bar
    fig.colorbar(p, ax=ax)

    # show and save plot to file
    #plt.show()
    # ax.figure.savefig('recall-work.png', dpi=300)
    return ax


def visualise_metric(metric):
    N = len(metric['x'][1])
    # normalise colours between 0-255
    colours = (np.arange(0, N)) / N * 255.0

    # format figure
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)
    ax.set_xlabel(metric['x'][0])
    ax.set_ylabel(metric['y'][0])

    # create distribution axes
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # create distribution plots
    p = scatter_hist(np.array(metric['x'][1]), np.array(metric['y'][1]), ax, colours, ax_histx, ax_histy)

    # create colour bar
    fig.colorbar(p, ax=ax)

    # show and return plot
    #plt.show()
    return ax


def scatter_hist(x, y, ax, colours, ax_histx, ax_histy):
    """
    Plots scatter plot with histograms showing distributions
    :param x: x-axis values
    :param y: y-axis values
    :param ax: axis object for plot
    :param colours: colour of the scatter plot
    :param ax_histx: axis for the x histogram
    :param ax_histy: axis for the y histogram
    :return: scatter plot
    """
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # plot main scatter data
    p = ax.scatter(x, y, c=colours, alpha=0.5)

    # determine limits for distribution axes
    binwidth = 0.01
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)

    # create distribution for x values, in y direction
    ax_histx.hist(x, bins=bins)

    # plot the mean of the x values on the main scatter plot
    plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(x.mean() * 1.1, max_ylim * 0.1, 'Mean: {:.2f}'.format(x.mean()))

    # create distribution for the y values, in x direction
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    # plot the mean of the y values on the main scatter plot
    plt.axhline(y.mean(), color='k', linestyle='dashed', linewidth=1)
    min_xlim, max_xlim = plt.xlim()
    plt.text(max_xlim * 0.7, y.mean() * 0.95, 'Mean: {:.2f}'.format(y.mean()))
    return p


def scatter_plot(metric, colour_label="index", marginal=True):
    title = metric['name']
    marginals = ""
    if marginal:
        marginals = "histogram"

    x_label = metric['x'][0]
    y_label = metric['y'][0]

    df = pd.DataFrame({x_label: metric['x'][1], y_label: metric['y'][1]})
    df = df.reset_index(level=0)

    fig = px.scatter(df, x=x_label, y=y_label, marginal_x=marginals, marginal_y=marginals, title=title, color=colour_label)
    #fig.update_traces(textposition='top center')
    return fig


def metric_plot(metric):
    title = metric['name']
    x_label = metric['x'][0]
    y_label = metric['y'][0]

    # plotly interactive plot
    df = pd.DataFrame({x_label: metric['x'][1], y_label: metric['y'][1]})
    df = df.reset_index(level=0)
    trace1 = px.line(df, x=x_label, y=y_label, title=title)

    # static plot
    # ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax = df.plot(kind='line', x=x_label, y=y_label, title=title)
    return ax
