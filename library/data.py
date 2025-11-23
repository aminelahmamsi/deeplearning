

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

class Dataset:
    
    def __init__(self) -> None:
        
        #Data preprocessing & augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        #Data preprocessing & augmentation for testing
        self.test_transform =transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #loading of the dataset
        data_dir = "./kaggle/input/brain-tumor/4 classes"
        self.dataset = ImageFolder(root=data_dir, transform=self.train_transform)
        
    def prepare_data(self, hyperParameters):
        
        # Split dataset (70-15-15 split)
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        
        #assign the preprocessing to the different sets
        val_dataset.dataset.transform = self.test_transform
        test_dataset.dataset.transform = self.test_transform
        
        #assigns the loaders to the sets
        train_loader = DataLoader(train_dataset, batch_size=hyperParameters.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=hyperParameters.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=hyperParameters.batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader