import argparse
import importlib
import os
import ssl
import sys

import numpy as np

from active_learner import ActiveLearner
from data_extraction import load_datasets, process_file_string, extract_datasets
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

    # perform TF-IDF feature extraction
    if not load:
        datasets = load_datasets(working_directory, 'data')
        for i, dataset in enumerate(datasets):
            datasets[i] = compute_TFIDF(dataset, 1000)
            datasets[i].to_pickle('./datasets/dataset_' + str(i) + '.pkl')
        print(datasets)
    # load precomputed TF-IDF dataset
    elif load:
        extract_datasets('datasets', working_directory)
        datasets = load_pkl_datasets(working_directory, 'datasets')
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
        data = {'train': datasets[i], 'dev': datasets[i]}
        (evaluator, stopper) = run_model(data, params)
        evaluators.append(evaluator)
        stoppers.append(stopper)
        recalls.append(evaluator.recall[-1])
        work_saves.append(evaluator.work_save[-1])
    print('Mean recall:', sum(recalls) / len(recalls))
    print('Mean work save:', sum(work_saves) / len(work_saves))

    visualise_training(evaluators[0], stoppers[0])
    visualise_results(evaluators)


# handle input arguments: specify model name + parameters
def handle_args():
    parser = argparse.ArgumentParser()

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
                        default=True)
    args = parser.parse_args()

    print(f"Args: {vars(args)}")

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

    return {'confidence': confidence,
            'model': (model_, model_params),
            'selector': (selector_, selector_params),
            'stopper': (stopper_, stopper_params),
            'evaluator': evaluator_}


def run_model(data, params):
    N = len(data['train'])
    batch_size = int(0.03 * N)

    model_AL = params['model'][0](*params['model'][1])
    selector = params['selector'][0](batch_size, params['confidence'], *params['selector'][1])
    stopper = params['stopper'][0](N, params['confidence'], *params['stopper'][1])
    #stopper = Statistical(N, 0.95, 0.95, verbose=True)

    evaluator = None
    if params['evaluator']:
        evaluator = params['evaluator'](data['train'])

    active_learner = ActiveLearner(model_AL, selector, stopper, batch_size=batch_size, max_iter=1000,
                                   evaluator=evaluator, verbose=False)

    (mask, relevant_mask) = active_learner.train(data['train'])

    recall = evaluator.recall[-1]
    print('Recall:', recall)
    work_save = evaluator.work_save[-1]
    print('Work save:', work_save)

    preds = model_AL.predict(data['dev'])
    y = data['dev']['y']
    print('Model predicted relevants:', sum(preds))

    print('Relevants found:', sum(relevant_mask))
    print('Actual number of relevants:', sum(y))
    print('Total reviews screened:', sum(mask))
    print('Total reviews:', N)

    print('\n')
    return evaluator, stopper


def load_pkl_datasets(working_directory, datasets_name):
    datasets = []
    for data_path in os.listdir(working_directory + datasets_name):
        (name, file_type) = process_file_string(data_path)
        if file_type == 'pkl':
            data = pd.read_pickle(datasets_name + '/' + name + '.' + file_type)
            datasets.append(data)
    return datasets


if __name__ == '__main__':
    main()
