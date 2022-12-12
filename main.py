import ssl
import pprint
import ssl

from active_learner import ActiveLearner
from command_line_interface import parse_CLI
from data_extraction import get_datasets
from evaluator import *
from stopper import *

working_directory = './'

load = True



def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    params = parse_CLI()
    pp = pprint.PrettyPrinter()
    print()
    pp.pprint(params)
    print()

    datasets = get_datasets(params['data'][0], params['data'][1], working_directory)
    #print(datasets)

    evaluators = []
    stoppers = []
    recalls = []
    work_saves = []

    # set randomisation seed
    np.random.seed(0)

    for i, dataset in enumerate(datasets):
        if i >= 100:
            break
        print("Analysing dataset {0} out of {1}...".format(i+1, len(datasets)+1))
        data = {'train': datasets[i], 'dev': datasets[i]}
        (evaluator, stopper) = run_model(data, params)
        evaluators.append(evaluator)
        stoppers.append(stopper)
        recalls.append(evaluator.recall[-1])
        work_saves.append(evaluator.work_save[-1])
    print('Mean recall:', sum(recalls) / len(recalls))
    print('Mean work save:', sum(work_saves) / len(work_saves))

    evaluator = evaluators[0]
    stopper = stoppers[0]
    metrics = [*(evaluator.get_eval_metrics()), *(stopper.get_eval_metrics())]

    visualise_training(metrics)
    visualise_results(evaluators)


def run_model(data, params):
    """
    Creates algorithm objects and runs the active learning program

    :param data: dataset for systematic review labelling
    :param params: input parameters for algorithms and other options for training
    :return: returns the evaluator and stopper objects trained on the dataset
    """
    N = len(data['train'])
    batch_size = int(0.03 * N)
    # TODO different batch sizes? as a parameter

    model_AL = params['model'][0](*params['model'][1])
    selector = params['selector'][0](batch_size, params['confidence'], *params['selector'][1], verbose=params['selector'][2])
    stopper = params['stopper'][0](N, params['confidence'], *params['stopper'][1], verbose=params['stopper'][2])

    evaluator = None
    if params['evaluator']:
        evaluator = params['evaluator'][0](data['train'], verbose=params['evaluator'][1])

    active_learner = ActiveLearner(model_AL, selector, stopper, batch_size=batch_size, max_iter=1000,
                                   evaluator=evaluator, verbose=params['active_learner'][1])

    (mask, relevant_mask) = active_learner.train(data['train'])

    evaluator.out(model_AL, data['dev'])
    return evaluator, stopper


if __name__ == '__main__':
    main()
