import torch.optim as optim
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import itertools
from MyDataset import MoviesDataset
from Net import Feedforward
from Train import train_model
from torch.utils.data import DataLoader, Subset
from MyUtility import weight_class, print_best_hyper_result, save_result
from Test import test_model
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    """Random seed for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    """ Hyperparameters """
    hidden_sizes = [16,32,64]
    nums_epochs = [500,800,1000]
    batch_sizes = [8,16,32,64]
    learning_rate = [0.1]
    momentum = [0,0.9]
    step_size_lr_decay = 100


    hyperparameters = itertools.product(hidden_sizes, nums_epochs, batch_sizes,
                                        learning_rate,momentum)
    ListofResult = []
    dataset = MoviesDataset()
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.25, stratify=dataset.y, random_state=16)

    """ Train and test model """
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, val_idx)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True)

    sampler = weight_class(dataset,train_subset)

    for hidden_size, num_epochs, batch, learning_rate, momentum in hyperparameters:

        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=False,
                                  drop_last=False, sampler=sampler)

        model = Feedforward(dataset.X.shape[1], hidden_size, dataset.num_classes)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_lr_decay, gamma=0.1)

        model, loss_values = train_model(model, criterion, optimizer, num_epochs,
                                         train_loader, device,scheduler,batch)

        ListofResult = test_model(model, test_loader,device,loss_values,ListofResult,
                                  dataset,hidden_size, num_epochs, batch, learning_rate, momentum)
    """ Print best hyperparameters """
    print_best_hyper_result(ListofResult)
    """ Save results """
    save_result(ListofResult)
