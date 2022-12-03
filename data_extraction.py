import re
import zipfile
import os
import pandas as pd


def load_datasets(working_directory, datasets_name):
    """
    Loads all datasets from zipped datasets file
    :param working_directory: path containing the datasets compressed file
    :param datasets_name: name of the compressed datasets file
    :return: list of pandas DataFrames containing all extracted datasets
    """
    extract_datasets(datasets_name, working_directory)
    datasets = []
    for data_path in os.listdir(working_directory + datasets_name):
        (name, file_type) = process_file_string(data_path)
        if file_type == 'csv':
            data = load_csv_data(working_directory + datasets_name + '/' + data_path, 'record_id',
                                 ['title', 'abstract'], 'label_included')
            data['x'] = data['title'] + data['abstract']
            data.rename(columns={'label_included': 'y'}, inplace=True)
            datasets.append(data)
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
    return None, None
