import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def read_dream_time_series(path_name, n):
    """Reads time series data from tsv file and converts data into numpy arrays of size (n, 11) where n represents the number
    of time points measured. The first column is the time point in seconds and the latter 10 columns represent gene 1 until gene 10. 
    There are 5 networks encoded in the file.

    See https://www.synapse.org/#!Synapse:syn3049712/wiki/74633 for more information.

    Args:
        path_name (str): path to data location
        n (int): number of time points (differs per data file)

    Returns:
        float[]: 5 timeseries of networks
    """
    data = pd.read_csv(path_name ,sep='\t')
    time_series = data.to_numpy()
    network1 = time_series[:n]
    network2 = time_series[n:2*n]
    network3 = time_series[2*n:3*n]
    network4 = time_series[3*n:4*n]
    network5 = time_series[4*n:5*n]
    return network1, network2, network3, network4, network5

def read_experiment_41(path_name):
    """Reads time series data from tsv file and outputs raw data.

    Args:
        path_name (str): path to data location
        
    Returns:
        float[]: timeseries of network
    """
    data = pd.read_csv(path_name ,sep='\t')
    time_series = data.to_numpy()
    return time_series


def read_yeast_data(path_name, genes_of_interest):
    data_to_save = {}
    with open(path_name, 'r') as file:
        for line in file.readlines():
            if any(gene in line for gene in genes_of_interest):
                line_separated = re.split(r'\t|\n', line)
                line_converted_to_float = np.array([float(i) for i in line_separated[3:len(line_separated)-1]])
                data_to_save[line_separated[0]] = line_converted_to_float

    return data_to_save

if __name__ == "__main__":
    genes_of_interest = ["YLR079W", "YPL256C", "YMR199W", "YPR119W", "	YDR146C", "YGR108W", "YNL068C"]
    data = read_yeast_data("data/yeast/Spellman_data_cdc.txt", genes_of_interest)
    
    plt.figure(figsize=(10, 7))
    time_points = [i for i in range(len(data[list(data.keys())[0]]))]
    all_data = np.array(list(data.values()))
    print(all_data)
    for key in data.keys():
        normalized_data = (data[key] - np.min(all_data)) / (np.max(all_data) - np.min(all_data))
        # normalized_data = (data[key]) / (np.linalg.norm(all_data, ord=2))
        plt.plot(time_points, normalized_data)

    plt.show()


