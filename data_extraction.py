import math
import pickle
import re
import warnings
import zipfile
import os

import numpy
import numpy as np
import pandas as pd
from tfidf import compute_TFIDF


# TODO: dont feature extract when dataset available. Feature extract save to directory for data

def get_datasets(data_path, data_name, data_file_type, max_datasets, output_directory, feature_extraction):
    """
    Get data from different formats and file types. Loads from .zip, directory with csv datas, directory with pkl datas

    :param data_path:
    :param data_name: name of the data directory or file
    :param data_file_type: string name of the data file type (None if directory)
    :param max_datasets:
    :param output_directory: working directory of the program
    :param feature_extraction:
    :return: list of datasets, including feature extractions (namely TF-IDF)
    """
    # extract compressed datasets
    if data_file_type == 'zip':
        extract_datasets(os.path.join(data_path, data_name), output_directory)
    # load each dataset
    datasets = []
    num_datasets = 0
    try:
        dataset_files = os.listdir(os.path.join(data_path, data_name))
    except FileNotFoundError:
        raise Exception("data not found") from None

    if feature_extraction:
        feature_extractor = feature_extraction[0](*feature_extraction[1])
    else:
        feature_extractor = None
        warnings.warn("no feature extraction specified")

    for path in dataset_files:
        if num_datasets - max_datasets == 0:
            break
        (p, name, file_type) = process_file_path(path)
        print("loaded:", name, file_type)
        # load csv dataset
        if file_type == '.csv':
            data = load_csv_data(os.path.join(data_path, data_name, name) + file_type, 'record_id', ['title', 'abstract'], 'label_included')
            data['x'] = data['title'] + data['abstract']
            data.rename(columns={'label_included': 'y'}, inplace=True)
        # load pkl dataset
        elif file_type == '.pkl':
            data = pd.read_pickle(os.path.join(data_path, data_name, name) + file_type)
        # skip unsupported data types
        else:
            warnings.warn("{name}.{type} has an unsupported data type".format(name=name, type=file_type))
            continue

        # compute TF-IDF feature representation
        if type(data.iloc[0]['x']) == str or type(data.iloc[0]['x']) == float:
            if feature_extractor:
                data = feature_extractor.extract_features(data)
            else:
                warnings.warn("dataset {name} required feature extraction but none was specified".format(name=name))
                continue

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


def process_file_path(file_string):
    """
    Processes file name, splits into name and file type

    :param file_string: name of the file
    :return: extracted name and file type
    """
    path, tail = os.path.split(file_string)
    name, file_type = os.path.splitext(tail)
    if path == '':
        path = "./"
    if file_type == '':
        file_type = None

    return path, name, file_type
