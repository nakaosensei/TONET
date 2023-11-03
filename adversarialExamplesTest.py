from scipy.io import arff
import json 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.TONetModel import *
from neuralNetwork.AttackerModel import *
from datasetLoaders.dualDatasetLoaderAdversarial import DualAdversarialDataSet
from datasetLoaders.datasetLoaderAdversarial import AdversarialDataSet
from torchmetrics import ConfusionMatrix
import os
import gc

f = open('settings.json')
settingsJson = json.load(f)
DEVICE=get_device()
datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
trainingPath='../savedModels/attackerModelWeka'
originalDatasetPath = '../outputs/originalDatabaseSamples/'
originalLabelsPath = '../outputs/originalDatabaseTargets/'
adversarialDatasetPath = '../outputs/adversarialExamples/'
adversarialLabelsPath = '../outputs/targets/'

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    #print(size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    confMatrix = ConfusionMatrix(task="MultiClass",num_classes=6)
    completeConfMatrix = None
    with torch.no_grad():
        for X, y in dataloader:            
            pred = model(X)           
            test_loss += loss_fn(pred, y).item()           
            results = confMatrix(pred.argmax(1),y)
            if completeConfMatrix is None:
                completeConfMatrix = results
            else:
                completeConfMatrix = torch.add(completeConfMatrix,results)

            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print('Matriz de confusao')    
    print(completeConfMatrix)
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return {'acuracia':100*correct,'matrizConfusao':completeConfMatrix}

def runTest(model):    
    results = ''
    articleBatchSize = settingsJson['batchSize']
    for i in range(0, len(settingsJson['testPercentages'])):
        print(str(settingsJson['testPercentages'][i])+'%'+' da base de amostras adversarias utilizado')
        if settingsJson['testPercentages'][i]==0:
            advDataset = AdversarialDataSet(originalDatasetPath,originalLabelsPath)
        else:
            advDataset = DualAdversarialDataSet(originalDatasetPath, originalLabelsPath,adversarialDatasetPath,adversarialLabelsPath, settingsJson['testPercentages'][i]) 
        preProcessed = advDataset.preProcessDataset()
        advDataset.loadDataset(preProcessed)
        loss_fn = nn.CrossEntropyLoss()
        test_dataloader = DataLoader(advDataset, batch_size=articleBatchSize, shuffle=True)
        testData = test(test_dataloader, model, loss_fn)
        accuracy = testData['acuracia']
        matrizConfusao = testData['matrizConfusao']
        save3DTensorAsStringFile(matrizConfusao,'../outputs/confusionMatrix/'+str(settingsJson['testPercentages'][i]))
        results += str(settingsJson['testPercentages'][i])+'%'+' da base de amostras adversarias utilizado\nAcurácia:'+str(accuracy)+'\n'
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
    model = AttackerNetwork()
    model.load_state_dict(torch.load(trainingPath))
    model.eval()    
    
    #Testa o modelo com base nos arquivos em featuresPath e targetsPath
    runTest(model)

    
    
    
    
    
