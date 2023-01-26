import math
import re
import zipfile
import os

import numpy
import numpy as np
import pandas as pd
from tfidf import compute_TFIDF


def get_datasets(data_name, data_file_type, working_directory, max_datasets):
    """
    Get data from different formats and file types. Loads from .zip, directory with csv datas, directory with pkl datas

    :param max_datasets:
    :param data_name: name of the data directory or file
    :param data_file_type: string name of the data file type (None if directory)
    :param working_directory: working directory of the program
    :return: list of datasets, including feature extractions (namely TF-IDF)
    """
    # extract compressed datasets
    if data_file_type == 'zip':
        extract_datasets(data_name, working_directory)
    # load each dataset
    datasets = []
    num_datasets = 0
    try:
        dataset_files = os.listdir(working_directory + data_name)
    except FileNotFoundError:
        raise Exception("data not found") from None

    for data_path in dataset_files:
        if num_datasets - max_datasets == 0:
            break
        (name, file_type) = process_file_string(data_path)
        print("loaded:", name, file_type)
        # load csv dataset
        if file_type == 'csv':
            data = load_csv_data(working_directory + data_name + '/' + data_path, 'record_id',
                                 ['title', 'abstract'], 'label_included')
            data['x'] = data['title'] + data['abstract']
            data.rename(columns={'label_included': 'y'}, inplace=True)
        # load pkl dataset
        elif file_type == 'pkl':
            data = pd.read_pickle(data_name + '/' + name + '.' + file_type)
        # skip unsupported data types
        else:
            print('WARNING:', file_type, 'is an unsupported data type')
            continue

        # compute TF-IDF feature representation
        if type(data.iloc[0]['x']) == str or type(data.iloc[0]['x']) == float:
            data = compute_TFIDF(data, 1000)

        datasets.append(data)
        num_datasets += 1

    return datasets



def extract_datasets(datasets_name, dest_path):
    """
    Extract zipped datasets

    :param datasets_name: file name of the datasets
    :param dest_path: destination file path to extract to
    """
    zip_path = datasets_name + '.zip'
    if not os.path.isdir(datasets_name):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)


def load_csv_data(data_path, index_name, feature_names, label_name):
    """
    Loads dataset from csv file

    :param data_path: file path of the dataset
    :param index_name: header name of the column to index with
    :param feature_names: list of header names of the columns to use as the features
    :param label_name: header name of the column to use as the ground truth labels (relevant or irrelevant)
    :return: pandas DataFrame of the dataset, including features 'x' and labels 'y'
    """
    data = pd.read_csv(data_path, index_col=index_name, usecols=[index_name, *feature_names, label_name])
    data.index = data.index - data.index[0]
    return data


def process_file_string(file_string):
    """
    Processes file name, splits into name and file type

    :param file_string: name of the file
    :return: extracted name and file type
    """
    pattern = "(^[\S\n\t\v ]*)[.]([a-z]*)$"
    regex_output = re.compile(pattern).split(file_string)
    if len(regex_output) >= 2:
        name = regex_output[1]
        file_type = regex_output[2]
        return name, file_type
    return file_string, None
