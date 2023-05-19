from scipy.io import arff
import json 
from classifiers.classifiers import *
from datasetLoaders.datasetLoader import TonetDataSet
f = open('settings.json')
settingsJson = json.load(f)

datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
trainingPath = '../savedModels/trainedAttackertst'
debugMode = False


if __name__=='__main__':    
    #To train a model, we need a loss function and an optimizer.    
    tonetDataset = TonetDataSet(datasets)    
    preProcessed = tonetDataset.preProcessDataset()
    runKnn(preProcessed['labels'], preProcessed['database'])    
    
    
    
    
