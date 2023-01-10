import argparse
import importlib
import os
from configparser import ConfigParser
import yaml

from data_extraction import process_file_string


# handle input arguments: specify model name + parameters
def parse_CLI():
    """
    Parses command line arguments into program parameters and algorithms
    :return: parameters for the program, model selector stopper evaluator etc.
    """
    # add optional arguments to look for
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Name of the config file (optional), overrides other command line arguments',
                        default='')
    args = parser.parse_args()

    params = []

    for filename in os.listdir(args.config):
        f = os.path.join(args.config, filename)
        # checking if it is a file
        if os.path.isfile(f):
            config_name, config_file_type = process_file_string(filename)
            if config_file_type == 'yml':
                data_args, algorithm_args, training_args = read_yml(f)
            elif config_file_type == 'ini':
                data_args, algorithm_args, training_args = read_ini(f)
            else:
                continue
        else:
            continue
        param = get_params(data_args, algorithm_args, training_args)
        param['name'] = config_name
        params.append(param)
    return params


def read_yml(config_file):
    """
    Reads a yaml configuration file and provides the relevant simulation parameters
    :param config_file: file path of the configuration file
    :return: data, algorithm, and training parameters
    """
    with open(config_file, 'r') as f:
        config_object = yaml.load(f, Loader=yaml.FullLoader)
    data_args = combine_dict_list(config_object["DATA"])
    algorithm_args = combine_dict_list(config_object["ALGORITHMS"])
    training_args = combine_dict_list(config_object["TRAINING"])

    data_args = process_config_args(data_args)
    algorithm_args = process_config_args(algorithm_args)
    training_args = process_config_args(training_args)
    return data_args, algorithm_args, training_args


def combine_dict_list(dlist):
    """
    Helper function for parsing yaml files. Combines list into single dictionary
    :param dlist: list of dictionaries
    :return: single flattened dictionary
    """
    result = {}
    for d in dlist:
        result.update(d)
    return result


def read_ini(config_file):
    """
    Reads an ini configuration file and provides the relevant simulation parameters
    :param config_file: file path of the configuration file
    :return: data, algorithm, and training parameters
    """
    config_object = ConfigParser()
    config_object.read(config_file)

    data_args = config_object["DATA"]
    algorithm_args = config_object["ALGORITHMS"]
    training_args = config_object["TRAINING"]

    data_args = process_config_args(data_args)
    algorithm_args = process_config_args(algorithm_args)
    training_args = process_config_args(training_args)
    return data_args, algorithm_args, training_args


def process_config_args(args):
    """
    Split argument string in list

    :param args
    :return: list of separate arguments
    """
    result = {}
    for key, val in args.items():
        result[key] = str(val).split(' ')
    return result


def get_params(data_args, algorithm_args, training_args):
    """
    Forms program parameters from arguments, including desired object classes and hyperparameters

    :param data_args: name of datasets directory or compressed file
    :param algorithm_args: names of methods and any desired parameters
    :param training_args: parameters for training, evaluator, and verbosity
    :return:
    """
    # specify dataset
    data_name, data_file_type = process_file_string(data_args['data'][0])

    # training hyper parameters
    confidence = float(training_args['confidence'][0])

    # machine learning model parameters
    model_args = algorithm_args['model']
    model_name = model_args[0]
    model_params = model_args[1:]
    model_module = importlib.import_module('model')
    model_ = getattr(model_module, model_name)

    # sample selection method parameters
    selector_args = algorithm_args['selector']
    selector_name = selector_args[0]
    selector_params = selector_args[1:]
    selector_module = importlib.import_module('selector')
    selector_ = getattr(selector_module, selector_name)

    # stopping criteria method parameters
    stopper_args = algorithm_args['stopper']
    stopper_name = stopper_args[0]
    stopper_params = stopper_args[1:]
    stopper_module = importlib.import_module('stopper')
    stopper_ = getattr(stopper_module, stopper_name)

    # evaluator, store for training results and metrics
    evaluator_module = importlib.import_module('evaluator')
    evaluator_ = getattr(evaluator_module, 'Evaluator')

    # active learner algorithm
    active_learner_module = importlib.import_module('active_learner')
    active_learner_ = getattr(active_learner_module, 'ActiveLearner')

    # verbosity specifications
    verbosity_args = training_args['verbose']
    model_verbosity = 'model' in verbosity_args
    selector_verbosity = 'selector' in verbosity_args
    stopper_verbosity = 'stopper' in verbosity_args
    evaluator_verbosity = True
    active_learner_verbosity = 'active_learner' in verbosity_args

    # compile parameters
    params = {'data': (data_name, data_file_type),
              'confidence': confidence,
              'model': (model_, model_params, model_verbosity),
              'selector': (selector_, selector_params, selector_verbosity),
              'stopper': (stopper_, stopper_params, stopper_verbosity),
              'evaluator': (evaluator_, evaluator_verbosity),
              'active_learner': (active_learner_, active_learner_verbosity)}
    return params
