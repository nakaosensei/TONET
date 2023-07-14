from scipy.io import arff
import numpy as np
import json 
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.TONetModel import *
import random

f = open('settings.json')
settingsJson = json.load(f)
labelsColumn = settingsJson['labelsColumn']


class AdversarialDataSet():

    def __init__(self,dataSetsPath,labelsPath):
        self.labels = None
        self.labelsStr = None
        self.labelsHashMap = {}
        self.tensorDatabase = None
        self.dataSetsPath=dataSetsPath
        self.labelsPath=labelsPath
        

    def upSampleClasses(self, dataset, classesHashMap):
        classesNames = list(classesHashMap.keys())
        maxSize = len(classesHashMap[classesNames[0]])
        for i in range(1, len(classesNames)):
            if len(classesHashMap[classesNames[i]])>maxSize:
                maxSize=len(classesHashMap[classesNames[i]])
        for i in range(0, len(classesNames)):
            while len(classesHashMap[classesNames[i]])<maxSize:
                pickedCopyIndex = random.randint(0,len(classesHashMap[classesNames[i]])-1)
                dataset.append(classesHashMap[classesNames[i]][pickedCopyIndex])                
                classesHashMap[classesNames[i]].append(classesHashMap[classesNames[i]][pickedCopyIndex])
        return [dataset, classesHashMap]

    def writeClassHashMap(self,classesHashMap):
        toWrite = {}
        for cl in classesHashMap.keys():
            toWrite[str(cl)]=len(classesHashMap[cl])
        
        with open('../inputs/'+'originalRegisters.json', 'w') as f:
            json.dump(toWrite, f)

    def writeUpSampledDatabase(self,filteredData):        
        with open('../data/TONet/'+'upscaledDatabase.json', 'w') as f:
            json.dump(filteredData, f)
    
    
    def convertVoidNoneTypesToEmptyStr(self, trafficInstance):
        convertedTrafficInstance = []
        distinctTypes = {}
        for i in range(0, len(trafficInstance)):
            distinctTypes[type(trafficInstance[i])]=trafficInstance[i]   
                
            if type(trafficInstance[i])==None:
                convertedTrafficInstance.append("")                
            elif type(trafficInstance[i])==bytes:
                try:
                    tmp = str(trafficInstance[i])
                    convertedTrafficInstance.append(tmp)
                except:
                    signedValue = []
                    for j in trafficInstance[i]:
                        if type(trafficInstance[j])==bytes:
                            signedValue.append(str(j))
                            continue
                        signedValue.append(j)
                    
                    convertedTrafficInstance.append(signedValue)
            else:
                convertedTrafficInstance.append(trafficInstance[i])       
        return convertedTrafficInstance

    def keepOnlyUsedColumns(self, trafficInstance):
        usedColumns = settingsJson['colunasConsideradasRawIndex']
        newInstance = []
        for columnIndex in usedColumns:
            newInstance.append(float(trafficInstance[int(columnIndex)]))
        newInstance.append(trafficInstance[-1])
        return newInstance
    
    def removeLabels(self,dataset):
        labels=[]        
        reducedDataset=[]
        for i in range(0,len(dataset)):
            if len(dataset[i])==0:
                emptyLines.append(i)
                continue
            labels.append(dataset[i][-1])
            instance=[]
            for j in range(0, len(dataset[i])-1):
                instance.append(dataset[i][j])
            reducedDataset.append(instance)
            #del dataset[i][-1]
        
        return {'labels':labels,'dataset':reducedDataset}

    def filterClasses(self,data,labels):
        classesHashMap = {} #key: classname value: array [] of registers
        outputDataset = []
        
        for i in range(0,len(data)):            
            className = labels[i]
            if className not in classesHashMap:
                classesHashMap[className]=[]
            classesHashMap[className].append(data[i])
                       
        for i in range(0,len(data)):                           
            outputDataset.append(data[i])        
        
        return {'database':outputDataset,'labels':labels,'classesHashMap':classesHashMap}

    def joinUDPColumns(self,totalData):
        for register in totalData:
            if str(register[-1])=="b'FTP-CONTROL'" or str(register[-1])=="b'FTP-PASV'" or str(register[-1])=="b'FTP-DATA'":
                register[-1]='Bulk (UDP)'

    def normalizeData(self, totalData):
        numpyTotalData = np.array(totalData)
        for i in range(0, len(totalData)):
            x = numpyTotalData[i]
            normalized = (x-np.min(x))/(np.max(x)-np.min(x)) 
            totalData[i]=normalized.tolist()
        return totalData

    def preProcessDataset(self):
        totalData = []
        labels = []
        print('Will load the datasets...')
        dataSetsName = os.listdir(self.dataSetsPath)        
        for datasetNm in dataSetsName:
            if '.pt' not in datasetNm or '.txt' in datasetNm:
                continue            
            tensor = torch.load(self.dataSetsPath+datasetNm)                   
            for register in tensor:                
                totalData.append(register)
                
        classesNames = os.listdir(self.labelsPath)
        for datasetNm in classesNames:
            if '.pt' not in datasetNm or '.txt' in datasetNm:
                continue
            tensor = torch.load(self.labelsPath+datasetNm)    
            for register in tensor:                
                labels.append(register)
        print('Raw Datasets loaded: COMPLETE')
        
        filteredData = self.filterClasses(totalData, labels)
        print('Filter classes: COMPLETE') 
        
                    
        print('Qt. registers on database:'+str(len(filteredData['database'])))
        
        return filteredData
   

    def loadDataset(self, preProcessed):            
        self.tensorDatabase = preProcessed['database']       

        tensorSizes = {}
        for tensor in self.tensorDatabase:
            tensorSizes[len(tensor)]=0
        
        self.labelsStr=preProcessed['labels']
        self.labelsHashMap = {}
        for i in range(0,len(self.labelsStr)):            
            if self.labelsStr[i].item() not in self.labelsHashMap:
                self.labelsHashMap[self.labelsStr[i].item()]=len(self.labelsHashMap.keys())                 
        
        self.labels = []
        for lb in self.labelsStr:
            self.labels.append(self.labelsHashMap[lb.item()])
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):        
        return self.tensorDatabase[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __calculate_std_mean__(self):
        return torch.std_mean(self.tensorDatabase, dim=0, keepdim=True)


def agregateTotals():
    files = ['entry1Registers.json','entry2Registers.json','entry3Registers.json','entry4Registers.json','entry5Registers.json','entry6Registers.json','entry7Registers.json','entry8Registers.json','entry9Registers.json','entry10Registers.json']
    totalJson={}
    for f in files:
        with open('../inputs/'+f, 'r') as f:
            data = json.load(f)
            for key in data.keys():
                if key not in totalJson:
                    totalJson[key]=data[key]
                    continue
                totalJson[key]+=data[key]
    with open('../inputs/agregado.json', 'w') as f:
        json.dump(totalJson, f)

def saveTensorAsStringFile(tensor,filename):
    out = ""
    for i in range(0, len(tensor)):
        for j in range(0,len(tensor[i])):
            if j==0:
                out+='['
            out+=str(float(tensor[i][j]))
            if j!=len(tensor[i])-1:
                out+=','
            else:
                out+=']'            
        out+='\n'
    f = open(filename,'w')
    f.write(out)
    f.close()

if __name__=='__main__':    
    datasetsPath = '../outputs/adversarialExamples/'
    labelsPath = '../outputs/targets/'
    advDataset = AdversarialDataSet(datasetsPath,labelsPath)
    preProcessed = advDataset.preProcessDataset()
    advDataset.loadDataset(preProcessed)
    train_dataloader = DataLoader(advDataset, batch_size=1000)
    train_features, train_labels = next(iter(train_dataloader))
    
    for (X, y) in enumerate(train_dataloader):
        print(X)
        print(y)
        exit()