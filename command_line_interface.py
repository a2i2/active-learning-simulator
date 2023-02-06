import argparse
import importlib
import importlib.util

from config import read_config_directory
from data_extraction import process_file_path


# handle input arguments: specify model name + parameters
def parse_CLI(argument_names):
    """
    Parses command line arguments into program parameters and algorithms

    :return: parameters for the program, model selector_algorithms stopper evaluator etc.
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
    """
    Handler for simulate.py parameters

    :param config_names: names of the configurations
    :param config_args: parsed configuration arguments with DATA, FEATURE EXTRACTION, MODEL, SELECTOR, STOPPER, TRAINING, OUTPUT
    :return: parameters object containing parsed parameters from config files
    """
    params = []
    for i, config_arg in enumerate(config_args):
        param = get_params(config_arg[0], config_arg[1], config_arg[2:5], config_arg[5], config_arg[6])
        param['name'] = config_names[i]
        params.append(param)
    return params


def create_clustering_params(config_names, config_args):
    """
    Handler for cluster_evaluation.py parameters

    :param config_names: names of the configurations
    :param config_args: parsed configuration arguments with DATA, FEATURE EXTRACTION, MODEL, SELECTOR, STOPPER, TRAINING, OUTPUT, CLUSTERING
    :return: parameters object containing parsed parameters from config files
    """
    params = []
    for i, config_arg in enumerate(config_args):
        param = get_params(config_arg[0], config_arg[1], config_arg[2:5], config_arg[5], config_arg[6])
        param['name'] = config_names[i]
        params.append(param)
        param['clusterer'] = get_clustering_params(config_arg[6])
    return params


def get_params(data_args, feature_args, algorithm_args, training_args, output_args):
    """
    Forms program parameters from arguments, including desired object classes and hyperparameters

    :param data_args: name of datasets directory or compressed file
    :param feature_args:
    :param algorithm_args: names of methods and any desired parameters
    :param training_args: parameters for training, evaluator, and verbosity
    :param output_args:
    :return: parsed parameters from the config arguments for active learning training
    """
    # specify dataset
    data_path, data_name, data_file_type = process_file_path(data_args['data'][0])
    data_number = -1
    if len(data_args['data']) > 1:
        data_number = int(data_args['data'][1])

    feature_params = None
    if feature_args['module'] != ['None'] and feature_args['class'] != ['None']:
        feature_params = get_algorithm_params_file(feature_args, 'feature extraction')
    print(feature_params)

    # training hyperparameters
    try:
        batch_proportion = float(training_args['batch proportion'][0])
    except ValueError:
        raise Exception("batch proportion must be a decimal value: decimal percentage of total dataset size") from None

    try:
        confidence = float(training_args['confidence'][0])
    except ValueError:
        raise Exception("confidence must be a decimal value") from None

    # machine learning model parameters
    model_params = get_algorithm_params_file(algorithm_args[0], 'model')
    # sample selection method parameters
    selector_params = get_algorithm_params_file(algorithm_args[1], 'selector')
    # stopping criteria method parameters
    stopper_params = get_algorithm_params_file(algorithm_args[2], 'stopper')

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
    output_path_args = output_args['output path'][0]
    if output_path_args == 'None':
        output_path_args = "./"
    output_metrics_args = output_args['output metrics']

    # compile parameters
    params = {'data': (data_path, data_name, data_file_type, data_number),
              'feature_extraction': feature_params,
              'batch_proportion': batch_proportion,
              'confidence': confidence,
              'model': model_params + (model_verbosity,),
              'selector': selector_params + (selector_verbosity,),
              'stopper': stopper_params + (stopper_verbosity,),
              'evaluator': (evaluator_, evaluator_verbosity),
              'active_learner': (active_learner_, active_learner_verbosity),
              'output_path': output_path_args + "/",
              'output_metrics': output_metrics_args}
    return params


def get_algorithm_params(algorithm_args, key):
    # sample selection method parameters
    args = algorithm_args[key]
    name = args[0]
    params = args[1:]
    module = importlib.import_module(key)
    try:
        class_ = getattr(module, name)
    except AttributeError:
        raise Exception("{key} could not be found".format(key=key)) from None
    return class_, params


def get_algorithm_params_file(args, key):
    """
    Parse config argument for retrieving a module class and optional parameter specification

    :param args: config arguments
    :param key: name of the key, e.g. 'model' or 'stopper'
    :return: class to instantiate, optional parameters
    """
    # sample selection method parameters
    module_name = args['module'][0]
    class_name = args['class'][0]
    params = args['parameters']
    if params == ['None'] or params == ['null']:
        params = []

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise Exception("{key} module could not be found: {name}".format(key=key, name=module_name)) from None

    try:
        class_ = getattr(module, class_name)
    except AttributeError:
        raise Exception("{key} could not be found: {name}".format(key=key, name=class_name)) from None
    return class_, params


def get_clustering_params(clustering_args):
    """
    Parse config argument for retrieving clusterer object

    :param clustering_args: config arguments for clustering evaluation
    :return: class to instantiate, optional parameters
    """
    clusterer_module = importlib.import_module('clusterer')
    try:
        clusterer_ = getattr(clusterer_module, clustering_args['clusterer'][0])
    except AttributeError:
        raise Exception("clusterer could not be found") from None
    params = (clusterer_, clustering_args['clusterer'][1:])
    return params
