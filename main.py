#!/usr/bin/env python3
import os
import pprint
import ssl

from active_learner import ActiveLearner
from command_line_interface import parse_CLI
from data_extraction import get_datasets
from evaluator import *
from stopper import *
from datetime import datetime

working_directory = './'

load = True


def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # set up output directory
    output_name = str(datetime.now())
    output_directory = working_directory + output_name
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # get desired parameters for training
    params = parse_CLI()
    pp = pprint.PrettyPrinter()
    print()
    pp.pprint(params)
    print()

    for param in params:
        # get datasets to train the program on
        full_datasets = get_datasets(param['data'][0], param['data'][1], working_directory)
        # truncate datasets for speed
        data_start = 0
        data_end = len(full_datasets)
        datasets = full_datasets[data_start: data_end]

        # store program objects for later evaluation
        active_learners = []
        recalls = []
        work_saves = []

        # set randomisation seed
        np.random.seed(0)

        output_string = ''
        # train for each dataset
        for i, dataset in enumerate(datasets):
            print("Analysing dataset {0} out of {1}...".format(i + 1, len(datasets)))
            data = {'train': datasets[i], 'dev': datasets[i]}
            active_learner = run_model(data, param)
            active_learners.append(active_learner)
            recalls.append(active_learner.evaluator.recall[-1])
            work_saves.append(active_learner.evaluator.work_save[-1])

            # show evaluation results
            output_string += 'Dataset {0} out of {1}...'.format(i + 1, len(datasets))
            output_string += active_learner.evaluator.out(active_learner.model, data['dev']) + '\n'

        output_string += 'Mean recall: {0}'.format(sum(recalls) / len(recalls))
        output_string += '\nMean work save: {0}'.format(sum(work_saves) / len(work_saves))
        print('Mean recall:', sum(recalls) / len(recalls))
        print('Mean work save:', sum(work_saves) / len(work_saves))
        print()

        output_file_name = output_directory + '/' + param['name']
        save_output_text(output_string, output_file_name)

        # visualise training results of a particular evaluator
        evaluator = active_learners[0].evaluator
        stopper = active_learners[0].stopper
        metrics = [*(evaluator.get_eval_metrics()), *(stopper.get_eval_metrics())]
        axs = visualise_training(metrics)

        # visualise the overall training results
        evaluators = [a.evaluator for a in active_learners]
        ax = visualise_results(evaluators)
        ax.figure.savefig(output_file_name + '_' + 'recall-work.png', dpi=300)


def run_model(data, params):
    """
    Creates algorithm objects and runs the active learning program

    :param data: dataset for systematic review labelling
    :param params: input parameters for algorithms and other options for training
    :return: returns the evaluator and stopper objects trained on the dataset
    """
    N = len(data['train'])
    # determine suitable batch size, batch size increases with increases dataset size
    batch_size = int(0.03 * N)
    # TODO different batch sizes? as a parameter

    # create algorithm objects
    model_AL = params['model'][0](*params['model'][1])
    selector = params['selector'][0](batch_size, params['confidence'], *params['selector'][1], verbose=params['selector'][2])
    stopper = params['stopper'][0](N, params['confidence'], *params['stopper'][1], verbose=params['stopper'][2])

    # specify evaluator object if desired
    evaluator = None
    if params['evaluator']:
        evaluator = params['evaluator'][0](data['train'], verbose=params['evaluator'][1])

    # TODO output just the active learner
    # create active learner
    active_learner = ActiveLearner(model_AL, selector, stopper, batch_size=batch_size, max_iter=1000,
                                   evaluator=evaluator, verbose=params['active_learner'][1])

    # train active learner
    (mask, relevant_mask) = active_learner.train(data['train'])

    return active_learner


def save_output_text(string, file_name):
    with open(file_name + '_results.txt', 'w') as f:
        f.write(string)


if __name__ == '__main__':
    main()
