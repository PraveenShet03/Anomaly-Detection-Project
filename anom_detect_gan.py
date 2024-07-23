import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pandas as pd
import pickle
import torch
from bayes_opt import BayesianOptimization
import json
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def get_measures(actual, pred, tol):
    """
      A function to measure, the total True Positives, False Negatives and False Positives for a given tolerence.
      Parameters
      --------------
      actual : a 1-d array
         Actual labels (1 or 0 entries)
      pred : a 1-d array of same size of actual
         Predicted labels (predicted labels)
      tol : int
         tolerance value used for calculation

      Returns
      ---------------
      TP: int
        Total true positives count
      FN: int
         Total false negatives count
      FP: int
         Total false positives count
    """
    TP = 0
    FN = 0
    FP = 0
    actual = np.array(actual)
    pred = np.array(pred)
    if len(pred) != 0:
        for a in actual:
            if min(abs(pred - a)) <= tol:
                TP = TP + 1
            else:
                FN = FN + 1
        for p in pred:
            if min(abs(actual - p)) > tol:
                FP = FP + 1
    else:
        FN = len(actual)

    return TP, FN, FP

# Evaluation function
def evaluate(test_df, test_out_dict, window_size, min_height, tol=24, thresh=0.5, alpha=1, beta=0, only_peaks=False):
    """
    A function to evaluate the f1 scores on the test dataset.
    Parameters
    -----------
      test_df: pandas dataframe
        The dataset obtained for testing (processed and includes "s_no" column).
      test_out_dict: Dictionary
          Dictionary containing reconstruction details with s_no (segment number) as keys.
      window_size:
          size of the subsequence (number of datapoints)
      min_height: float in range (0,1)
         Setting threshold on the KDE curve
      thresh: float
        threshold on the reconstruction error to identify critical subsequences.
      alpha, beta: floats
         anomaly score params
      only_peaks: bool
        If true only the peaks in the region of KDE above min-height are marked as anomalous timestamps
        else, all the timestamps in those regions are marked as anomalies.
    """
    TP = 0
    FN = 0
    FP = 0
    temp = test_df.groupby("s_no")
    for id, id_df in temp:
        id_df.reset_index(drop=True, inplace=True)
        id_dict = test_out_dict[id]
        error = id_dict["recon_loss"]
        z_norm = id_dict["Z"]
        if type(error) == torch.Tensor:
            error = error.detach().cpu().numpy()
        z_norm = torch.norm(z_norm.view(-1, lat_dim), dim=1).detach().cpu().numpy()
        combined_score = alpha * error + beta * z_norm
        mask = combined_score > thresh
        if not id_dict["window_b_included"]:
            mask = np.pad(mask, (window_size // 2 - 1,), mode='constant', constant_values=False)

        print(len(id_df)-len(mask), id_dict["window_a_included"], id_dict["window_b_included"])

        positions = np.where(mask)[0]
        if len(positions) <= 1:
            anom = positions
        else:
            kde = gaussian_kde(positions, bw_method=0.05)
            x = range(0, len(id_df))
            y = kde(x)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

            if only_peaks:
                peaks, _ = find_peaks(y, height=min_height)
            else:
                peaks = np.where(y > min_height)[0]

            anom = peaks
        actual_anom = id_df.index[id_df['anomaly'] == 1]
        TP_i, FN_i, FP_i = get_measures(actual=actual_anom, pred=anom, tol=tol)
        TP = TP + TP_i
        FN = FN + FN_i
        FP = FP + FP_i
    return TP, FN, FP

def k_fold_evaluation(df, test_out_dict, window_size, min_height, tol=24, thresh=0.5, alpha=1, beta=0, only_peaks=False, k=5):
    kf = KFold(n_splits=k)
    all_tps, all_fns, all_fps = [], [], []
    all_actuals, all_preds = [], []
    for train_index, test_index in kf.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        TP, FN, FP = evaluate(test_df, test_out_dict, window_size, min_height, tol, thresh, alpha, beta, only_peaks)
        all_tps.append(TP)
        all_fns.append(FN)
        all_fps.append(FP)
        
        # Collect actual and predicted values for ROC
        for id, id_df in test_df.groupby("s_no"):
            id_df.reset_index(drop=True, inplace=True)
            id_dict = test_out_dict[id]
            error = id_dict["recon_loss"]
            z_norm = id_dict["Z"]
            if type(error) == torch.Tensor:
                error = error.detach().cpu().numpy()
            z_norm = torch.norm(z_norm.view(-1, lat_dim), dim=1).detach().cpu().numpy()
            combined_score = alpha * error + beta * z_norm
            mask = combined_score > thresh
            if not id_dict["window_b_included"]:
                mask = np.pad(mask, (window_size // 2 - 1,), mode='constant', constant_values=False)

            positions = np.where(mask)[0]
            if len(positions) <= 1:
                anom = positions
            else:
                kde = gaussian_kde(positions, bw_method=0.05)
                x = range(0, len(id_df))
                y = kde(x)
                y = (y - np.min(y)) / (np.max(y) - np.min(y))

                if only_peaks:
                    peaks, _ = find_peaks(y, height=min_height)
                else:
                    peaks = np.where(y > min_height)[0]

                anom = peaks
            actual_anom = id_df['anomaly'].values
            actual_anom_scores = np.zeros_like(actual_anom)
            actual_anom_scores[anom] = 1
            all_actuals.extend(actual_anom)
            all_preds.extend(actual_anom_scores)
    
    return np.mean(all_tps), np.mean(all_fns), np.mean(all_fps), np.array(all_actuals), np.array(all_preds)

def plot_roc_auc(actual, pred_scores, b_id, tol):
    fpr, tpr, _ = roc_curve(actual, pred_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Building {b_id} (tol={tol})')
    plt.legend(loc="lower right")
    
    # Ensure the directory exists
    os.makedirs(f'plots/roc', exist_ok=True)
    plt.savefig(f'plots/roc/roc_curve_{b_id}_tol_{tol}.png')
    plt.close()

if __name__ == "__main__":

    with open('config.json', 'r') as file:
        config = json.load(file)

    window_size = config['preprocessing']['window_size']
    iters = config["recon"]["iters"]
    use_dtw = config["recon"]["use_dtw"]
    eval_mode = config["recon"]["use_eval_mode"]
    lat_dim = config['training']['latent_dim']

    tolerance = [12, 24]

    df = pd.read_csv(config["data"]["dataset_path"])
    b_ids = df["building_id"].unique()
    del df
    print(f"unique builds : {b_ids}")

    results_df = pd.DataFrame(
        columns=['b_id', 'use_dtw', 'alpha', 'beta', 'thresh', 'min_height', 'Precision', 'Recall', 'F1','tol'])

    for b_id in b_ids:
        print(b_id)
        start_time = time.time()
        for dtw in [use_dtw]:

            b_df = pd.read_csv(f"dataset/test_df_{b_id}.csv")
            with open(f"test_out/iters_{iters}_reconstruction_{b_id}_{dtw}_{eval_mode}.pkl", "rb") as f:
                test_out_dict = pickle.load(f)

            def black_box_function(thresh, min_height, alpha, beta):
                TP, FN, FP = evaluate(b_df, test_out_dict, window_size, min_height, 6, thresh, alpha, beta)
                try:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                    F1 = 2 * P * R / (P + R)
                except:
                    F1 = 0
                return F1

            pbounds = {'thresh': (0, 100), 'min_height': (0.4, 1), "alpha": (0, 1), "beta": (0, 1)}

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=1, allow_duplicate_points=True
            )

            optimizer.maximize(init_points=70, n_iter=180)

            thresh = optimizer.max["params"]["thresh"]
            min_height = optimizer.max["params"]["min_height"]
            alpha = optimizer.max["params"]["alpha"]
            beta = optimizer.max["params"]["beta"]

            for tol in tolerance:
                TP, FN, FP, actual, preds = k_fold_evaluation(b_df, test_out_dict, window_size, min_height, tol, thresh, alpha, beta)
                P = TP / (TP + FP)
                R = TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                results_df.loc[len(results_df)] = [b_id, dtw, alpha, beta, thresh, min_height, P, R, F1, tol]
                
                # Plot and save ROC curve
                plot_roc_auc(actual, preds, b_id, tol)

        end_time = time.time()
        print(f"Time take for building {b_id} is {end_time-start_time}")

    if eval_mode:
        t1 = "eval_mode_on"
    else:
        t1 = "eval_mode_off"

    if use_dtw:
        t2 = "soft_dtw"
    else:
        t2 = "mse"

    results_df.to_csv(f"results_{t1}_{t2}.csv")