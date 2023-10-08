from scipy.io import arff
import json 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.TONetModel import *
from datasetLoaders.dualDatasetLoaderAdversarial import DualAdversarialDataSet
from datasetLoaders.datasetLoader import TonetDataSet
import os
import gc

f = open('settings.json')
settingsJson = json.load(f)
DEVICE=get_device()
datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
trainingPath='../savedModels/trainedTonet'
originalDatasetPath = '../outputs/originalDatabaseSamples/'
originalLabelsPath = '../outputs/originalDatabaseTargets/'
adversarialDatasetPath = '../outputs/adversarialExamples/'
adversarialLabelsPath = '../outputs/targets/'
labelsPercentage = 10

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
    return 100*correct

def runTest(model):    
    results = ''
    articleBatchSize = settingsJson['batchSize']
    for i in range(0, len(settingsJson['testPercentages'])):
        advDataset = DualAdversarialDataSet(originalDatasetPath, originalLabelsPath,adversarialDatasetPath,adversarialLabelsPath, settingsJson['testPercentages'][i]) 
        preProcessed = advDataset.preProcessDataset()
        advDataset.loadDataset(preProcessed)
        loss_fn = nn.CrossEntropyLoss()
        test_dataloader = DataLoader(advDataset, batch_size=articleBatchSize, shuffle=True)
        accuracy = test(test_dataloader, model, loss_fn)
        results += str(settingsJson['testPercentages'][i])+'por cento de amostras adversarias\nAcurácia:'+str(accuracy)+'\n'
        del test_dataloader
        del loss_fn
        del advDataset
        del preProcessed
        gc.collect()
    f = open('tests.txt','w')
    f.write(results)
    f.close()




'''
Basta executar o script stochasticAdversarialTester.py para testar
as amostras adversariais geradas com um modelo pre-treinado.
'''
if __name__=='__main__':    
    #Carrega o modelo pré-treinado  
    model = ToNetNeuralNetwork()
    model.load_state_dict(torch.load(trainingPath))
    model.eval()    

    
    #Testa o modelo com base nos arquivos em featuresPath e targetsPath
    runTest(model)

    
    
    
    
    
