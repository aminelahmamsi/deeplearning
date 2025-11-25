import itertools
import torch

from library.data import Dataset
from library.hyper_parameters import HyperParameters

from library.get_device import get_device
from library.get_id import get_or_create_unique_id
from library.training import train_model
from library.evaluation import evaluate
from library.csv_processing import save_training_results_to_csv


def automeasure(runs_per_measure=5, no_print=True):
    
    learning_rates = [0.005, 0.001, 0.0002]
    dropouts = [0.5, 0.2]
    optimizer_types = ["adam", "sgd"]
    batch_sizes = [32]
    epochs = [20]
    
    param_combinations = itertools.product(
        learning_rates, batch_sizes, epochs, dropouts, optimizer_types
    )

    device = get_device()
    device_id = get_or_create_unique_id()
    
    dataset = Dataset()
    num_combinations = (
        len(learning_rates) *
        len(batch_sizes) *
        len(epochs) *
        len(dropouts) *
        len(optimizer_types) *
        runs_per_measure
    )
    current_run = 0

    print(f"starting the {num_combinations} runs measurements!")
    
    for lr, bs, ep, dr, op in param_combinations:
        
        for i in range(runs_per_measure):
            current_run += 1
            
            result = {}
            hyperParameters = HyperParameters(lr, bs, ep, dr, op)
            
            #Initialize the model, criterion, optimizer:
            model = hyperParameters.build_model(device)
            criterion = hyperParameters.build_criterion()
            optimizer = hyperParameters.build_optimizer(model)
            train_loader, val_loader, test_loader = dataset.prepare_data(hyperParameters)
            
            #preparing the data: storing the hyperparameters's values
            result_init = {
                    'computer_id': device_id,
                    'learning_rate': hyperParameters.learning_rate,
                    'batch_size': hyperParameters.batch_size,
                    'epochs': hyperParameters.epochs,
                    'dropout': hyperParameters.dropout,
                    'optimizer_type': hyperParameters.optimizer_type,
                }
            
            #preparing the data: training of the model & storing results
            results_training = train_model(model, train_loader, val_loader, criterion, optimizer, device, hyperParameters, no_print)
            
            #preparing the data: testing of the model & storing results
            results_evaluation = evaluate(model, test_loader, device, no_print)
            
            #saving the data: storing it in the csv
            save_training_results_to_csv(result_init | results_training | results_evaluation)
            # Save the Model
            #torch.save(model.state_dict(), f"./models/model_{results_training["run_id"]}.pth")
            print(f"run {current_run}/{num_combinations} completed, id:{results_training["run_id"]}")
    
    print(f"all {num_combinations} runs have been completed, this program has finished successfully!")