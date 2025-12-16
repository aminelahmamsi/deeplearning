import torch.nn as nn
import torch.optim as optim

from library.model import BrainTumorCNN

#Initialize the hyperparameters:
class HyperParameters():
    def __init__(self,
                 learning_rate = 0.001,
                 batch_size = 32,
                 epochs = 10,
                 dropout = 0.5,
                 optimizer_type = "adam"
                 ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer_type = optimizer_type
        
    def build_model(self, device):
        return BrainTumorCNN(num_classes=4, dropout=self.dropout).to(device)

    def build_optimizer(self, model):
        if self.optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: '{self.optimizer_type}'. "
                         f"Supported types: 'adam', 'sgd'.")

    def build_criterion(self):
        return nn.CrossEntropyLoss()
