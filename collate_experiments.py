#!/usr/bin/env python3

import argparse
import json
import os

from evaluator import scatter_plot


def collate_experiments():
    """
    Compile together the results from different experiments (executions of the simulate.py program), visualises config comparison.

    :return:
    """
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Name of the config file directory',
                        default='./outputs')
    parser.add_argument('name', help='Shared name of the output files',
                        default='overall.json')
    args = parser.parse_args()
    directory = args.directory
    name = args.name

    collated_dict = {}
    config_names = []

    # traverse directory for desired output json file
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(name):
                continue
            file_path = os.path.join(root, file)
            config_name = os.path.basename(root)
            config_names.append(config_name)

            with open(file_path) as json_file:
                data = json.load(json_file)
                # parse each metric in the output data, collate values together
                for metric in data:
                    for key, value in metric.items():
                        if key in collated_dict:
                            collated_dict[key].append(value)
                        else:
                            collated_dict[key] = [value]

    # form and plot metrics
    metric_min = {'name': 'min metrics', 'x': ('work save', collated_dict['min_work_save']), 'y': ('recall', collated_dict['min_recall'])}
    ax = scatter_plot(metric_min, colour_label=config_names, marginal=False, text=config_names)
    ax.write_html("{path}/{fig_name}.html".format(path=directory, fig_name="min metrics"))

    metric_mean = {'name': 'mean metrics', 'x': ('work save', collated_dict['mean_work_save']), 'y': ('recall', collated_dict['mean_recall'])}
    ax = scatter_plot(metric_mean, colour_label=config_names, marginal=False, text=config_names)
    ax.write_html("{path}/{fig_name}.html".format(path=directory, fig_name="mean metrics"))
    return


if __name__ == '__main__':
    collate_experiments()
