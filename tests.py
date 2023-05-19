from scipy.io import arff
import json 
import pandas as pd
import torch
from cwLibrary.cw import L2Adversary
from neuralNetwork.lossFunctions import *
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.TONetModel import *
from datasetLoaders.datasetLoader4D import TonetDataSet


f = open('settings.json')
settingsJson = json.load(f)
DEVICE=get_device()
datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
trainingPath='../savedModels/trainedTonet'

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):        
        # Compute prediction error
        #print(X)
        #print(len(X))
        #print(len(X[0]))
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
    #print(size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
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
    meanStd = tonetDataset.__calculate_std_mean__()
    mean = torch.unsqueeze(meanStd[0],dim=0)
    std = torch.unsqueeze(meanStd[1],dim=0) 
    
    #loss_fn = distanceLoss2Norm
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize, shuffle=True)
    test_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize, shuffle=True)
    epochs = articleEpochs
    #epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)        
        runCw2(model, train_dataloader, meanStd[0], meanStd[1])
        
    torch.save(model.state_dict(), trainingPath)
    print("Done!")  
    
    return model

def runTest(model):
    articleBatchSize = settingsJson['batchSize']
    tonetDataset = TonetDataSet(datasets)    
    preProcessed = tonetDataset.preProcessDataset()
    tonetDataset.loadDataset(preProcessed)
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize, shuffle=True)
    test(test_dataloader, model, loss_fn)

def mountInputsBox(mean,std):
    zipped = zip(mean, std)    
    menor = None
    maior = None
    i = 0
    for m, s in zipped:
        tmpMin = (0 - m) / s
        tmpMax = (1 - m) / s         
        if i == 0:
            menor = tmpMin
            maior = tmpMax
        else:
            if menor>tmpMin:
                menor=tmpMin
            if maior<tmpMax:
                maior=tmpMax
        i+=1    
    return (min(menor),max(maior))    

def runCw2(net, dataloader, mean, std):    
    inputs_box = mountInputsBox(mean,std)
    #inputs_box = (min((0 - m) / s for m, s in zipped),max((1 - m) / s for m, s in zipped))
    # an untargeted adversary
    print('box')
    print(inputs_box)
    adversary = L2Adversary(targeted=False,confidence=0.0,search_steps=10,box=inputs_box, optimizer_lr=1e-3)

    inputs, targets = next(iter(dataloader))
    outputs = net(torch.autograd.Variable(inputs))
    print('inputs')
    print(inputs)
    #inputs = torch.unsqueeze(inputs,dim=0)
    #inputs = torch.unsqueeze(inputs,dim=0)
    print(len(inputs.size()))
    print('outputs')
    print(type(outputs))
    print(outputs)
    
    print('targets')
    print(len(targets.size()))

    adversarial_examples = adversary(net, inputs, targets, to_numpy=False)
    assert isinstance(adversarial_examples, torch.FloatTensor)
    assert adversarial_examples.size() == inputs.size()

    # a targeted adversary
    adversary = L2Adversary(targeted=True,confidence=0.0,search_steps=10,box=inputs_box,optimizer_lr=1e-3)
    inputs, _ = next(iter(dataloader))
    # a batch of any attack targets
    attack_targets = torch.ones(inputs.size(0)) * 3
    adversarial_examples = adversary(net, inputs, attack_targets, to_numpy=False)
    assert isinstance(adversarial_examples, torch.FloatTensor)
    assert adversarial_examples.size() == inputs.size()


def testAdvxs_var(model):
    tensor = torch.load('inputs.pt')
    out = ""
   
    
    tanhFunction = torch.nn.Tanh()
    output = tanhFunction(tensor)
    for i in range(0, len(output)):
       for j in range(0,len(output[i])):
           out+=str(float(output[i][j]))+' | '
       out+='\n'
    print(out)
    f = open('tahnFunctionNN.txt','w')
    f.write(out)
    f.close()
    exit()
    for i in range(0,len(pred)):
        for j in range(0,len(pred[i])):
            print(float(pred[i][j]))
    return model

if __name__=='__main__':    
    #To train a model, we need a loss function and an optimizer.    
    model = ToNetNeuralNetwork()
    model.load_state_dict(torch.load(trainingPath))
    model.eval()    
    testAdvxs_var(model)
    #runTraining(model)
    #runTest(model)

    
    
    
    
    
