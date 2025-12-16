import os
import csv
import pandas as pd
import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

def save_training_results_to_csv(results, save_path='training_results.csv'):
    """
    Save neural network training results to a CSV file.
    
    Parameters:
    -----------
    results : dict
        The results produced by training the neural network
    save_path : str, optional
        Path to save the CSV file (default: 'training_results.csv')
    
    Returns:
    --------
    str
        Path to the saved CSV file
    """
    
    #might add confusion table later

    results['epoch'] = list(range(1, results['epochs'] + 1))
    # Create DataFrame
    df = pd.DataFrame(results)
    
    #rounds values:
    df = df.round(4)
    
    # Check if file exists
    file_exists = os.path.isfile(save_path)
    
    # Write to CSV
    if file_exists:
        # Append without writing headers
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        df.to_csv(save_path, index=False)
    
    return save_path

# Example usage:
# results = train_network_enhanced_weighted(...)
# save_training_results_to_csv(results, "ResNet18_CIFAR10")

def process_results_csv():
    # Load CSV
    df = pd.read_csv(
        "training_results.csv",
        header=0,        # first row is column names
        dtype=str,       # IMPORTANT: prevents type-alignment corruption
        na_filter=True,  # normal NA handling
        low_memory=False # safer column inference on large files
        )

    # Keep only the last epoch of each run_id
    last_epochs = (
        df.sort_values("epoch")
        .groupby("run_id")
        .tail(1)
    )
    
    #normalize data:
    scaler = MinMaxScaler()
    last_epochs[["learning_rate", "dropout"]] = scaler.fit_transform(
        last_epochs[["learning_rate", "dropout"]]
    )

    
    last_epochs["test_acc"] = pd.to_numeric(last_epochs["test_acc"], errors="coerce")
    
    last_epochs["test_acc_bc"], lambda_bc = stats.boxcox(last_epochs["test_acc"])
    
    # Remove the now useless columns
    last_epochs.drop(['epoch','computer_id'], axis=1, inplace=True)

    # Save the result
    last_epochs.to_csv("cleaned_results.csv", index=False)

    print("Successfully processed results!")