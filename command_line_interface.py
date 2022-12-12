import argparse
import importlib
from configparser import ConfigParser

from data_extraction import process_file_string


# handle input arguments: specify model name + parameters
def parse_CLI():
    """
    Parses command line arguments into program parameters and algorithms
    :return: parameters for the program, model selector stopper evaluator etc.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Name of the config file (optional), overrides other command line arguments',
                        default='')
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

    # read from config file, override other command line arguments
    if args.config != '':
        config_object = ConfigParser()
        config_object.read(args.config)

        data_args = config_object["DATA"]
        algorithm_args = config_object["ALGORITHMS"]
        training_args = config_object["TRAINING"]
    # read from command line arguments
    else:
        data_args = args.data
        algorithm_args = {'model': args.model, 'selector': args.selector, 'stopper': args.stopper}
        training_args = {'confidence': args.confidence, 'evaluator': args.evaluator, 'verbose': args.verbose}
    params = get_params(data_args, algorithm_args, training_args)
    return params


def process_config_arguments(argument):
    """
    Split argument string in list
    :param argument:
    :return: list of separate arguments
    """
    return argument.split(' ')



def get_params(data_args, algorithm_args, training_args):
    """
    
    :param data_args:
    :param algorithm_args:
    :param training_args:
    :return:
    """
    data_name, data_file_type = process_file_string(data_args['data'])

    confidence = float(training_args['confidence'])

    model_args = process_config_arguments(algorithm_args['model'])
    model_name = model_args[0]
    model_params = model_args[1:]
    model_module = importlib.import_module('model')
    model_ = getattr(model_module, model_name)

    selector_args = process_config_arguments(algorithm_args['selector'])
    selector_name = selector_args[0]
    selector_params = selector_args[1:]
    selector_module = importlib.import_module('selector')
    selector_ = getattr(selector_module, selector_name)

    stopper_args = process_config_arguments(algorithm_args['stopper'])
    stopper_name = stopper_args[0]
    stopper_params = stopper_args[1:]
    stopper_module = importlib.import_module('stopper')
    stopper_ = getattr(stopper_module, stopper_name)

    evaluator = training_args['evaluator']
    evaluator_module = importlib.import_module('evaluator')
    if evaluator:
        evaluator_ = getattr(evaluator_module, 'Evaluator')
    else:
        evaluator_ = None

    active_learner_module = importlib.import_module('active_learner')
    active_learner_ = getattr(active_learner_module, 'ActiveLearner')

    verbosity_args = process_config_arguments(training_args['verbose'])
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