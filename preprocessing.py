import pandas as pd
import numpy as np
import json
from utils import plot_segments
import torch


def impute_nulls(data):
    """
    Mean for all the buildings is used for imputing (This could be improved later).
    Parameter
    ---------
      data: pandas dataframe
       A dataset containing columns: ['building_id','meter_reading'].
    Returns
    --------
      data: pandas dataframe
        A dataframe with missing meter readings imputed.
    """
    mean_reading = data.groupby('building_id').mean()['meter_reading']
    building_id = mean_reading.index
    values = mean_reading.values

    for i, idx in enumerate(building_id):
        data[data['building_id'] == idx] = data[data['building_id'] == idx].fillna(values[i])
    return data


def preprocess_data(data, dropna=True):
    """
    Preprocessing the input dataframe. By default dropping missing value is used.
    Users could add more preprocessing steps.
    Parameters
    ------------
       data: pandas dataframe
         A dataset containing columns: ['building_id','meter_reading',"timestamp"].
       dropna: bool
        If true, missing meter readings are simply removed, else missing meter readings are imputed
        using the method "impute_nulls".
    Returns
    -----------
       data: pandas dataframe
         A preprocessed dataset ready for further processing.

    """
    data = data.sort_values(by="timestamp")
    print(f'unique buildings : {data["building_id"].unique()}')
    if dropna:
        data.dropna(subset=['meter_reading'], inplace=True)
    else:
        data = impute_nulls(data)
    return data


def segment_data(data, n_segments=25, normalize=True):
    """
    A function which will divide the datasets in anomalous and normal segments. (train/test)
    Parameters
    -------------
        data: pandas dataframe
          Dataset with columns "building_id","anomaly" and "meter reading"
          Each timestamp is annotated with 1/0 for being an anomaly/normal timestamp and stored in the column anomaly.
        n_segments: int
          Number of segments in which the timeseries data for each building  should be divided.
        normalize: bool
        If true, each segment is normalize to lie in the range [-1,1]
    Returns
    ------------
        Processed pandas dataframes with additional column "s_no".
        Indicating the segment that a particular entry belongs to.

        normal_df: pandas dataframe
          The dataset with no anomalies which is used for training.
        ano_df : pandas dataframe
          The dataset with segments which contains at least one anomalous timestamp.
        s_no : int
          Returns the total number of segments in the dataset.
        min_seg_len: int
          The number of datapoints (timestamps) contained in the smallest segment.
    """
    min_seg_len = len(data)
    temp = data.groupby("building_id")
    normal_df = pd.DataFrame()  # train, valid (all normal entries)
    ano_df = pd.DataFrame()  # test (mixed)
    s_no = 0
    for id, id_df in temp:  # The column remains in id_df
        segments_per_build = np.array_split(id_df, n_segments)
        for d in segments_per_build:
            if len(d) < min_seg_len:
                min_seg_len = len(d)
            s_no = s_no + 1
            d["s_no"] = s_no
            if normalize == True:
                seq_x = d["meter_reading"]
                if np.max(seq_x) - np.min(seq_x) == 0:
                    seq_x = np.zeros(seq_x.shape)
                else:
                    seq_x = 2 * (seq_x - np.min(seq_x)) / (np.max(seq_x) - np.min(seq_x)) - 1
                d["meter_reading"] = seq_x

            if 1 in d['anomaly'].values:
                ano_df = pd.concat([ano_df, d])
            else:
                normal_df = pd.concat([normal_df, d])

    del temp

    return normal_df, ano_df, s_no, min_seg_len  # s_no to get the total number of segments


def split_sequence(sequence, n_steps, centering=False, minmax=False):
    """
    A function which will create the inputs for the models by passing a sliding window with 1 step size.
    Parameters
    -------------
        sequence: 1-D array
           Typically the entire meter readings from a segment in chronological order.
        n_steps: int
           Widow size for creating a subsequence
        centering: bool
           if true then standard normalization is applied to each subsequence.
        minmax: bool
           if true min-max scaling is done on the subsequence so that each reading
           within the subsequence is between the range of [-1,1].
   Returns
   -----------
        X: numpy array
           A 2-D array with each row as a subsequence.
    """
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_x = np.array(seq_x)
        if centering == True:
            std = np.std(seq_x)
            if std == 0:
                seq_x = np.zeros(seq_x.shape)
            else:
                seq_x = (seq_x - np.mean(seq_x)) / std
        if minmax == True:
            if np.max(seq_x) - np.min(seq_x) == 0:
                seq_x = np.zeros(seq_x.shape)
            else:
                seq_x = 2 * (seq_x - np.min(seq_x)) / (np.max(seq_x) - np.min(seq_x)) - 1

        X.append(seq_x)
    X = np.array(X)
    return X


if __name__ == "__main__":

    # Load config
    with open('config.json', 'r') as file:
        config = json.load(file)

    window_size = config['preprocessing']['window_size']
    b_id = "all" # default value, all the buildings in the dataset are taken together.

    # train-test segments
    data = pd.read_csv(f"{config['data']['dataset_path']}")
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]   # one particular building at a time (recommended)
        data = data[data["building_id"] == b_id]

    data = preprocess_data(data)
    train_df, test_df, s_no, min_len = segment_data(data)
    print(f"total number of segments : {s_no}")
    print(f" min len of segment: {min_len}")

    # plotting segments
    if config["preprocessing"]["plot_segments"]:
        plot_segments(train_df)
        plot_segments(test_df,False)

    # storing segments
    train_df.to_csv(f"dataset/train_df_{b_id}.csv", index=False)
    test_df.to_csv(f"dataset/test_df_{b_id}.csv", index=False)  # will be used for testing later

    # Convert training data into model input:
    X_train = []
    seg_count = 0
    temp = train_df.groupby("s_no")
    for id, id_df in temp:
        X_w = split_sequence(id_df["meter_reading"], window_size)
        X_train.extend(X_w)
        seg_count += 1
    X_train = np.array(X_train)
    X_train = X_train.reshape(len(X_train), 1, -1)

    print(f"training tensor shape: {X_train.shape}")
    torch.save(X_train,  config["data"]["train_path"] + f"X_train_{b_id}.pt" )
    print(f'The model training input is stored at : {config["data"]["train_path"]}')
