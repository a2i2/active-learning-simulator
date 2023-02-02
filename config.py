#!/usr/bin/env python3

import argparse
import itertools
import os
import warnings
from itertools import cycle

import numpy as np
import yaml

from data_extraction import process_file_string


def create_configs_combinations():
    # create options
    # data format: "data_directory num_datasets_to_screen"
    data = [["datasets"]]

    # model format: ("module_name", "class_name")
    model = [("model_algorithms.NB", "NB"),
             ("model_algorithms.LR", "LR"),
             ("model_algorithms.SVC", "SVC"),
             ("model_algorithms.MLP", "MLP")]
    # selector format: ("module_name", "class_name")
    selector = [("selector_algorithms.highest_confidence", "HighestConfidence"),
                ("selector_algorithms.lowest_entropy", "LowestEntropy"),
                ("selector_algorithms.weighted_sample", "WeightedSample")]
    # stopper format: ("module_name", "class_name")
    stopper = [("stopper_algorithms.consecutive_count", "ConsecutiveCount"),
               ("stopper_algorithms.sample_proportion", "SampleProportion"),
               ("stopper_algorithms.statistical", "Statistical")]

    # batch_proportions format: list of floats between 0 and 1 (not inclusive)
    batch_proportions = list(np.linspace(0.01, 0.05, 10))
    # confidence format: list of floats between 0 and 1 (not inclusive)
    confidence = list(np.linspace(0.8, 0.99, 10))
    # verbosity format: list objects to enable verbosity for (see documentation for available verbosity)
    verbosity = [("selector stopper")]

    # output format: ("output_path", "desired metrics separated by spaces")
    output = [("outputs", "true_recall model_recall stopper selector model")]

    # combine training parameters: cycle the smallest list of parameters
    training = list(zip(batch_proportions, confidence, cycle(verbosity)))

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-directory', help='Name of the config file directory',
                        default='./configs_generated')
    args = parser.parse_args()

    # create output directory
    directory = args.directory
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # restrict options to whatever is desired
    data = data[:]
    model = model[:]
    selector = selector[0:1]
    stopper = stopper[:]
    training = training[:]
    output = output[:]

    # form every possible combination of different parameters
    combinations = list(itertools.product(*[data, model, selector, stopper, training, output]))
    print("Generating configs:", len(combinations), "combinations")

    for i, combo in enumerate(combinations):
        param = generate_yml(combo)
        with open("{path}/config-{name}.yml".format(path=directory, name=i), 'w') as yaml_file:
            yaml.dump(param, yaml_file, default_flow_style=False)


def generate_yml(combo):
    param = {}
    param['DATA'] = [{'data': combo[0][0]}]
    param['MODEL'] = [{'module': combo[1][0]}, {'class': combo[1][1]}, {'parameters': None}]
    param['SELECTOR'] = [{'module': combo[2][0]}, {'class': combo[2][1]}, {'parameters': None}]
    param['STOPPER'] = [{'module': combo[3][0]}, {'class': combo[3][1]}, {'parameters': None}]
    param['TRAINING'] = [{'batch proportion': float(combo[4][0])}, {'confidence': float(combo[4][1])},
                         {'verbose': combo[4][2]}]
    param['OUTPUT'] = [{'output path': combo[5][0]}, {'output metrics': combo[5][1]}]
    return param




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
    else:
        warnings.warn("Config file {config} is not supported, try yaml".format(config=config_name))
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

