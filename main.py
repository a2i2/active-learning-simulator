import argparse
import importlib
import os
import ssl
import sys

import numpy as np

from active_learner import ActiveLearner
from data_extraction import process_file_string, get_datasets
from evaluator import *

from model import *
from selector import *
from stopper import *

from tfidf import compute_TFIDF

working_directory = './'

load = True



def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    params = handle_args()
    print(params)

    datasets = get_datasets(params['data'][0], params['data'][1], working_directory)
    print(datasets)

    evaluators = []
    stoppers = []
    recalls = []
    work_saves = []

    # set randomisation seed
    np.random.seed(0)

    for i, dataset in enumerate(datasets):
        if i >= 100:
            break
        print("Analysing dataset {0} out of {1}...".format(i, len(datasets)))
        data = {'train': datasets[i], 'dev': datasets[i]}
        (evaluator, stopper) = run_model(data, params)
        evaluators.append(evaluator)
        stoppers.append(stopper)
        recalls.append(evaluator.recall[-1])
        work_saves.append(evaluator.work_save[-1])
    print('Mean recall:', sum(recalls) / len(recalls))
    print('Mean work save:', sum(work_saves) / len(work_saves))

    visualise_training(evaluators[-1], stoppers[-1])
    visualise_results(evaluators)


# handle input arguments: specify model name + parameters
def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', help='Name of datasets directory',
                        default='datasets')
    parser.add_argument('--confidence', type=float, help='Confidence level, target recall',
                        default=0.95)
    parser.add_argument("--model", help="Base machine learning model: <model name> <model parameters (optional)>",
                        default='NB', nargs='+')
    parser.add_argument("--selector",
                        help="Selection criteria algorithm: <selector name> <selector parameters (optional)",
                        default='HighestConfidence', nargs='+')
    parser.add_argument("--stopper", help="Stopping criteria algorithm: <stopper name> <stopper parameters (optional)",
                        default='Statistical', nargs='+')
    parser.add_argument("--evaluator",
                        help="True or false, evaluation object for storing statistics and presenting detailed results",
                        default=True, nargs='*')
    parser.add_argument("--verbose", help="Specify which subsystems should produce verbose outputs",
                        default='evaluator', nargs='*')
    args = parser.parse_args()

    data_name, data_file_type = process_file_string(args.data)

    confidence = args.confidence

    model_name = args.model[0]
    model_params = args.model[1:]
    model_module = importlib.import_module('model')
    model_ = getattr(model_module, model_name)

    selector_name = args.selector[0]
    selector_params = args.selector[1:]
    selector_module = importlib.import_module('selector')
    selector_ = getattr(selector_module, selector_name)

    stopper_name = args.stopper[0]
    stopper_params = args.stopper[1:]
    stopper_module = importlib.import_module('stopper')
    stopper_ = getattr(stopper_module, stopper_name)

    evaluator = args.evaluator
    evaluator_module = importlib.import_module('evaluator')
    if evaluator:
        evaluator_ = getattr(evaluator_module, 'Evaluator')
    else:
        evaluator_ = None

    active_learner_module = importlib.import_module('active_learner')
    active_learner_ = getattr(active_learner_module, 'ActiveLearner')

    verbosity_args = args.verbose
    model_verbosity = 'model' in verbosity_args
    selector_verbosity = 'selector' in verbosity_args
    stopper_verbosity = 'stopper' in verbosity_args
    evaluator_verbosity = 'evaluator' in verbosity_args
    active_learner_verbosity = 'active_learner' in verbosity_args

    params = {'data': (data_name, data_file_type),
              'confidence': confidence,
              'model': (model_, model_params, model_verbosity),
              'selector': (selector_, selector_params, selector_verbosity),
              'stopper': (stopper_, stopper_params, stopper_verbosity),
              'evaluator': (evaluator_, evaluator_verbosity),
              'active_learner': (active_learner_, active_learner_verbosity)}
    return params


def run_model(data, params):
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
