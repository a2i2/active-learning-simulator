import matplotlib.pyplot as plt
import numpy as np


class Evaluator:
    def __init__(self, data):
        self.n = [0]
        self.r_AL = [0]
        self.r_total = sum(data['y'])
        self.recall = [0]
        self.N = len(data)
        self.N_AL = [0]
        self.k = [0]
        self.work_save = [0.0]
        self.tau_model = [0.0]

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


def visualise_training(evaluator, stopper):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    colour = 'tab:red'
    ax1.set_xlabel('Documents seen')
    ax1.set_ylabel('Recall')
    ax1.plot(evaluator.N_AL, evaluator.recall, label='Recall')
    ax1.tick_params(axis='y')
    fig1.tight_layout()
    ax1.set_title('Recall during training')
    legend = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    colour2 = 'tab:blue'
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('p-value')
    ax2.plot(range(len(stopper.ps)), stopper.ps, label='p-values')
    ax2.tick_params(axis='y')
    fig2.tight_layout()
    ax2.set_title('p-values during training')
    legend = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    colour3 = 'tab:green'
    ax3.set_xlabel('Documents seen')
    ax3.set_ylabel('Recall')
    ax3.plot(evaluator.N_AL[len(evaluator.N_AL) - len(evaluator.tau_model):], evaluator.tau_model, label='Model recall')
    ax3.tick_params(axis='y')
    fig3.tight_layout()
    ax3.set_title('Model recall during training')
    legend = ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    ax1.figure.savefig('recall.png', dpi=300)
    ax2.figure.savefig('p-value.png', dpi=300)
    ax3.figure.savefig('model-recall.png', dpi=300)


def visualise_results(evaluators):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Recall - work save')
    ax1.set_xlabel('Work save')
    ax1.set_ylabel('Recall')

    recalls = np.zeros(shape=(len(evaluators), 1))
    work_saves = np.zeros(shape=(len(evaluators), 1))
    colours = np.zeros(shape=(len(evaluators), 1))
    N_min = evaluators[0].N
    N_max = evaluators[0].N

    for i, evaluator in enumerate(evaluators):
        recalls[i] = evaluator.recall[-1]
        work_saves[i] = evaluator.work_save[-1]
        colours[i, :] = [evaluator.N]  # , evaluator.N, evaluator.N]
        N_min = min(evaluator.N, N_min)
        N_max = min(evaluator.N, N_max)

    # normalsie colours
    colours = (colours - N_min) / N_max * 255.0
    p = ax1.scatter(work_saves, recalls, c=colours, alpha=0.5)

    fig1.colorbar(p, ax=ax1)

    plt.show()
    ax1.figure.savefig('recall-work.png', dpi=300)