import os
import csv
import pandas as pd
import numpy as np
import torch

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