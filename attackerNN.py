from scipy.io import arff
import json 
import pandas as pd
import torch
from neuralNetwork.lossFunctions import *
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.AttackerModel import *
from datasetLoader import TonetDataSet
f = open('settings.json')
settingsJson = json.load(f)
DEVICE=get_device()
datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
trainingPath = '../savedModels/attackedModel'
debugMode = False

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):        
        # Compute prediction error
        if debugMode:
            print('X Instancias: len(X)'+str(len(X)))
            print('len(X[0]):'+str(len(X[0])))
            print(X)
            print('len(y):'+str(len(y)))
            print(y)
        pred = model(X)        
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print(size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #print('pred')
            #print(pred[0])
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def runTraining(model):
    model = model.to(DEVICE)
    articleBatchSize = settingsJson['batchSize']
    articleEpochs = settingsJson['epochs']
    
    tonetDataset = TonetDataSet(datasets)    
    preProcessed = tonetDataset.preProcessDataset()
    tonetDataset.loadDataset(preProcessed)
    #loss_fn = distanceLoss2Norm
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize, shuffle=True)
    test_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize, shuffle=True)
    epochs = 50
    #epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    torch.save(model.state_dict(), trainingPath)
    print("Done!")  

def runTest(model):
    model.load_state_dict(torch.load(trainingPath))
    articleBatchSize = settingsJson['batchSize']
    tonetDataset = TonetDataSet(datasets)    
    preProcessed = tonetDataset.preProcessDataset()
    tonetDataset.loadDataset(preProcessed)
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize, shuffle=True)
    test(test_dataloader, model, loss_fn)


if __name__=='__main__':    
    #To train a model, we need a loss function and an optimizer.    
    model = AttackerNetwork()
    
    #model.eval()    
    runTraining(model)
    #runTest(model)
    
    
    
    
    
