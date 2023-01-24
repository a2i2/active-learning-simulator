#!/usr/bin/env python3
import os
import pprint
import ssl

from active_learner import ActiveLearner
from command_line_interface import parse_CLI, create_simulator_params
from data_extraction import get_datasets
from evaluator import *
from stopper import *
from datetime import datetime

working_directory = './'
#TODO add 95% recall line to plots


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
    arg_names, args = parse_CLI(["DATA", "ALGORITHMS", "TRAINING"])
    params = create_simulator_params(arg_names, args)

    pp = pprint.PrettyPrinter()
    print()
    pp.pprint(params)
    print()

    mean_recalls = []
    min_recalls = []
    mean_work_saves = []
    min_work_saves = []

    # for each configuration
    for param in params:
        pp.pprint(param)
        print()
        # get datasets to train the program on
        datasets = get_datasets(param['data'][0], param['data'][1], working_directory, param['data'][2])

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

        mean_recall = sum(recalls) / len(recalls)
        mean_recalls.append(mean_recall)

        min_recall = min(recalls)
        min_recalls.append(min_recall)

        mean_work_save = sum(work_saves) / len(work_saves)
        mean_work_saves.append(mean_work_save)

        min_work_save = min(work_saves)
        min_work_saves.append(min_work_save)

        print('Mean recall:', mean_recall)
        print('Minimum recall:', min_recall)
        print('Mean work save:', mean_work_save)
        print('Minimum work save:', min_work_save)
        print()

        output_path = output_directory + '/' + param['name']
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        save_output_text(output_string, output_path, "results.txt")

        # visualise training results of a particular evaluator
        evaluator = active_learners[0].evaluator
        stopper = active_learners[0].stopper
        metrics = [*(evaluator.get_eval_metrics()), *(stopper.get_eval_metrics())]
        axs = visualise_training(metrics)

        # visualise the overall training results
        evaluators = [a.evaluator for a in active_learners]
        ax = visualise_results(evaluators)
        ax.figure.savefig(output_path + '/recall-work.png', dpi=300)

        output_results(active_learners, output_path)

    # config metrics
    ax = visualise_configs(mean_work_saves, mean_recalls)
    ax.figure.savefig(output_directory + '/configs_' + 'mean-recall-work.png', dpi=300)
    ax = visualise_configs(min_work_saves, min_recalls)
    ax.figure.savefig(output_directory + '/configs_' + 'min-recall-work.png', dpi=300)

    print('Configs mean recall:', sum(mean_recalls) / len(mean_recalls))
    print('Configs minimum recall:', min(min_recalls))
    print('Configs mean work save:', sum(mean_work_saves) / len(mean_work_saves))
    print('Configs minimum work save:', min(min_work_saves))
    print()
    #TODO add overview / summary for config results / ouputting file


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
    evaluator = params['evaluator'][0](data['train'], verbose=params['evaluator'][1])

    # create active learner
    active_learner = ActiveLearner(model_AL, selector, stopper, batch_size=batch_size, max_iter=1000,
                                   evaluator=evaluator, verbose=params['active_learner'][1])

    # train active learner
    (mask, relevant_mask) = active_learner.train(data['train'])

    return active_learner



def save_output_text(string, output_path, file_name):
    with open("{path}/{name}".format(path=output_path, name=file_name), 'w') as f:
        f.write(string)


if __name__ == '__main__':
    main()
