import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import matplotlib.pyplot as plt
import copy
from model import CNNClassif, init_weights
from data_prep import ImageDataset
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('braille_cnn')
ex.observers.append(FileStorageObserver('runs'))


@ex.config
def config():
    """Configuration of the Braille Image Classifier experiment."""
    seed = 0
    batch_size = 8
    num_epochs = 30
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.01
    num_channels1 = 16
    num_channels2 = 32
    num_channels3 = 64


@ex.capture
def training_cnn_classifier(model, train_dataloader, val_dataloader, num_epochs, loss_fn, learning_rate, verbose=True):
    model_tr = copy.deepcopy(model)
    model_tr.train()
    
    optimizer = torch.optim.SGD(model_tr.parameters(), lr=learning_rate)
    
    loss_all_epochs = []
    
    for epoch in range(num_epochs):
        loss_current_epoch = 0
        
        for batch_index, (images, labels) in enumerate(train_dataloader):
            
            y_pred = model_tr.forward(images)
            loss = loss_fn(y_pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_current_epoch += loss.item()

        loss_all_epochs.append(loss_current_epoch / (batch_index + 1))
        val_accuracy = eval_cnn_classifier(model_tr, eval_dataloader=val_dataloader)
        if verbose:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_current_epoch/(batch_index + 1):.4f}')
            print(f'-----> Validation Accuracy: {val_accuracy:.3f}%')
            ex.log_scalar('loss', loss_current_epoch, step=epoch+1)
        
    return model_tr, loss_all_epochs


def eval_cnn_classifier(model, eval_dataloader):

    model.eval() 

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in eval_dataloader:
            y_predicted = model(images)
            _, label_predicted = torch.max(y_predicted.data, 1)
            total += labels.size(0)
            correct += (label_predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    return accuracy


@ex.automain
def run(seed, batch_size, num_epochs, loss_fn, learning_rate, num_channels1, num_channels2, num_channels3):
    # Instantiating the dataset
    dataset = ImageDataset()
    # Splitting the dataset
    split_data = random_split(dataset, [1248, 156, 156], generator=torch.Generator().manual_seed(seed))
    train_data, val_data, test_data = split_data

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    num_classes = len(list(set([datapoint[1] for datapoint in train_data])))
    print("Number of classes: ", num_classes)
    batch_data, batch_name =  next(iter(train_dataloader))
    print(f'Batch shape [batch_size, image_shape]: {batch_data.shape}')
    print('Number of batches:', len(train_dataloader))

    print("== Initializing model...")
    model = CNNClassif(num_channels1, num_channels2, num_channels3, num_classes)
    torch.manual_seed(seed)
    model.apply(init_weights)
    num_params = sum(p.numel() for p in model.parameters())
    ex.log_scalar('number_of_params', num_params)
    print(model)

    print("== Training...")
    model, loss_total = training_cnn_classifier(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'test_model.pt')
    ex.add_artifact('test_model.pt')

    # TO DO: make it prettier
    plt.plot(list(range(num_epochs)), loss_total)
    plt.savefig('loss.png')
    ex.add_artifact('loss.png')

    print("== Evaluating...")
    accuracy = eval_cnn_classifier(model, test_dataloader)
    ex.log_scalar('accuracy', accuracy)
    return f'{round(accuracy, 2)}%'
