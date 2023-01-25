import argparse
import importlib
import os
from configparser import ConfigParser
import yaml
import warnings

from data_extraction import process_file_string






class CLIParser:
    def __init__(self):
        pass






# handle input arguments: specify model name + parameters
def parse_CLI(argument_names):
    """
    Parses command line arguments into program parameters and algorithms
    :return: parameters for the program, model selector stopper evaluator etc.
    """
    # add optional arguments to look for
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Name of the config file or directory',
                        default='')
    parser.add_argument('-other', help='Other parameters',
                        default='')
    args = parser.parse_args()

    config_names, config_args = read_config_directory(args.config, argument_names)
    return config_names, config_args


def create_simulator_params(config_names, config_args):
    params = []
    for i, config_arg in enumerate(config_args):
        param = get_params(config_arg[0], config_arg[1], config_arg[2], config_arg[3])
        param['name'] = config_names[i]
        params.append(param)
    return params


def create_clustering_params(config_names, config_args):
    params = []
    for i, config_arg in enumerate(config_args):
        param = get_params(config_arg[0], config_arg[1], config_arg[2], config_arg[3])
        param['name'] = config_names[i]
        params.append(param)
        param['clusterer'] = get_clustering_params(config_arg[4])
    return params


def read_config_directory(directory, argument_names):
    config_names = []
    config_args = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            config_name, config_file_type = process_file_string(filename)
            args = read_config(f, config_name, config_file_type, argument_names)
            if args:
                config_args.append(args)
                config_names.append(config_name)
        else:
            continue
    return config_names, config_args


def read_config(config_file, config_name, config_type, argument_names):
    args = None
    if config_type == 'yml':
        args = read_yml(config_file, argument_names)
    elif config_type == 'ini':
        args = read_ini(config_file, argument_names)
    else:
        warnings.warn("Config file {config} is not supported".format(config=config_name))
    return args


def read_yml(config_file, argument_names):
    """
    Reads a yaml configuration file and provides the relevant simulation parameters
    :param argument_names:
    :param config_file: file path of the configuration file
    :return: data, algorithm, and training parameters
    """
    with open(config_file, 'r') as f:
        config_object = yaml.load(f, Loader=yaml.FullLoader)
    args = []
    for arg_name in argument_names:
        args.append(process_config_args(combine_dict_list(config_object[arg_name])))
    return args


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


def read_ini(config_file, argument_names):
    """
    Reads an ini configuration file and provides the relevant simulation parameters
    :param argument_names:
    :param config_file: file path of the configuration file
    :return: data, algorithm, and training parameters
    """
    config_object = ConfigParser()
    config_object.read(config_file)
    args = []
    for arg_name in argument_names:
        args.append(process_config_args(config_object[arg_name]))
    return args


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


def get_params(data_args, algorithm_args, training_args, output_args):
    """
    Forms program parameters from arguments, including desired object classes and hyperparameters

    :param output_args:
    :param data_args: name of datasets directory or compressed file
    :param algorithm_args: names of methods and any desired parameters
    :param training_args: parameters for training, evaluator, and verbosity
    :return:
    """
    # specify dataset
    data_name, data_file_type = process_file_string(data_args['data'][0])
    data_number = -1
    if len(data_args['data']) > 1:
        data_number = int(data_args['data'][1])

    # training hyperparameters
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

    # output specifications
    working_directory_args = output_args['working path'][0]
    if working_directory_args == 'None':
        working_directory_args = "./"
    output_path_args = output_args['output path'][0]
    if output_path_args == 'None':
        output_path_args = working_directory_args
    output_metrics_args = output_args['output metrics']


    # compile parameters
    params = {'data': (data_name, data_file_type, data_number),
              'confidence': confidence,
              'model': (model_, model_params, model_verbosity),
              'selector': (selector_, selector_params, selector_verbosity),
              'stopper': (stopper_, stopper_params, stopper_verbosity),
              'evaluator': (evaluator_, evaluator_verbosity),
              'active_learner': (active_learner_, active_learner_verbosity),
              'working_path': working_directory_args,
              'output_path': output_path_args,
              'output_metrics': output_metrics_args}
    return params


def get_clustering_params(clustering_args):
    clusterer_module = importlib.import_module('clusterer')
    clusterer_ = getattr(clusterer_module, clustering_args['clusterer'][0])
    params = (clusterer_, clustering_args['clusterer'][1:])
    return params
