#!/usr/bin/env python3

import argparse
import itertools
import os
import warnings
from itertools import cycle

import numpy as np
import yaml

from data_extraction import process_file_path


def create_configs_combinations():
    # create options
    # data format: "data_directory num_datasets_to_screen"
    data = [["datasets"]]

    feature_extraction = [("tfidf", "TFIDF", 10000)]

    # model format: ("module_name", "class_name")
    model = [("model_algorithms.NB", "NB", None),
             ("model_algorithms.LR", "LR", None),
             ("model_algorithms.SVC", "SVC", None),
             ("model_algorithms.MLP", "MLP", None)]
    # selector format: ("module_name", "class_name")
    selector = [("selector_algorithms.highest_confidence", "HighestConfidence", None),
                ("selector_algorithms.lowest_entropy", "LowestEntropy", None),
                ("selector_algorithms.weighted_sample", "WeightedSample", None)]
    # stopper format: ("module_name", "class_name")
    stopper = [("stopper_algorithms.consecutive_count", "ConsecutiveCount", None),
               ("stopper_algorithms.sample_proportion", "SampleProportion", None),
               ("stopper_algorithms.statistical", "Statistical", None)]

    # batch_proportions format: list of floats between 0 and 1 (not inclusive)
    batch_proportions = list(np.linspace(0.01, 0.05, 10))
    # confidence format: list of floats between 0 and 1 (not inclusive)
    confidence = list(np.linspace(0.8, 0.99, 10))
    # verbosity format: list objects to enable verbosity for (see documentation for available verbosity)
    verbosity = [(None)]

    # output format: ("output_path", "desired metrics separated by spaces")
    output = [("outputs", "true_recall model_recall stopper selector model")]

    # combine training parameters: cycle the smallest list of parameters
    training = list(zip(batch_proportions, confidence, cycle(verbosity)))

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Name of the config file directory',
                        default='./configs_generated')
    args = parser.parse_args()

    # create output directory
    directory = args.directory
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # restrict options to whatever is desired
    data = data[:]
    feature_extraction = feature_extraction[:]
    model = model[:]
    selector = selector[0:1]
    stopper = stopper[:]
    training = training[:]
    output = output[:]

    # form every possible combination of different parameters
    combinations = list(itertools.product(*[data, feature_extraction, model, selector, stopper, training, output]))
    print("Generating configs:", len(combinations), "combinations")

    for i, combo in enumerate(combinations):
        param = generate_yml(combo)
        with open("{path}/config-{name}.yml".format(path=directory, name=i), 'w') as yaml_file:
            yaml.dump(param, yaml_file, default_flow_style=False)


def generate_yml(combo):
    param = {}
    param['DATA'] = [{'data': combo[0][0]}]
    param['FEATURE EXTRACTION'] = [{'module': combo[1][0]}, {'class': combo[1][1]}, {'parameters': combo[1][2]}]
    param['MODEL'] = [{'module': combo[2][0]}, {'class': combo[2][1]}, {'parameters': combo[2][2]}]
    param['SELECTOR'] = [{'module': combo[3][0]}, {'class': combo[3][1]}, {'parameters': combo[2][2]}]
    param['STOPPER'] = [{'module': combo[4][0]}, {'class': combo[4][1]}, {'parameters': combo[2][2]}]
    param['TRAINING'] = [{'batch proportion': float(combo[5][0])}, {'confidence': float(combo[5][1])}, {'verbose': combo[5][2]}]
    param['OUTPUT'] = [{'output path': combo[6][0]}, {'output metrics': combo[6][1]}]
    return param


def read_config_directory(directory, argument_names):
    config_names = []
    config_args = []

    # singular config file
    arg_path, arg_name, arg_type = process_file_path(directory)
    if arg_type:
        args, config_name = read_config(directory, argument_names)
        if args:
            config_args.append(args)
            config_names.append(config_name)
    # config directory
    else:
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            args, config_name = read_config(f, argument_names)
            if args:
                config_args.append(args)
                config_names.append(config_name)
    return config_names, config_args


def read_config(f, argument_names):
    # checking if it is a file
    if os.path.isfile(f):
        config_path, config_name, config_file_type = process_file_path(f)
        args = None
        if config_file_type == '.yml':
            args = read_yml(f, argument_names)
        else:
            warnings.warn("Config file {config} is not supported, try .yml".format(config=config_name))
        if args:
            return args, config_name
    return None, None


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


if __name__ == '__main__':
    create_configs_combinations()

