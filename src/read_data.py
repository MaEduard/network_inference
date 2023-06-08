import pandas as pd
import os

def read_dream_time_series(path_name, n):
    """Reads time series data from tsv file and converts data into numpy arrays of size (n, 11) where n represents the number
    of time points measured. The first column is the time point in seconds and the latter 10 columns represent gene 1 until gene 10. 
    There are 5 networks encoded in the file.

    See https://www.synapse.org/#!Synapse:syn3049712/wiki/74633 for more information.

    Args:
        path_name (str): path of data location
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
