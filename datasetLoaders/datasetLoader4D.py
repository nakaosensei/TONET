from scipy.io import arff
import numpy as np
import json 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.TONetModel import *
import random

f = open('settings.json')
settingsJson = json.load(f)
labelsColumn = settingsJson['labelsColumn']


class TonetDataSet():

    def __init__(self,dataSetsName):
        self.labels = None
        self.labelsStr = None
        self.labelsHashMap = {}
        self.tensorDatabase = None
        self.dataSetsName=dataSetsName
        

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

    def filterClasses(self,arffLoadedData):
        classesHashMap = {} #key: classname value: array [] of registers
        outputDataset = []
        for i in range(0, len(arffLoadedData)):
            arffLoadedData[i] = self.keepOnlyUsedColumns(arffLoadedData[i].tolist())
            arffLoadedData[i] = self.convertVoidNoneTypesToEmptyStr(arffLoadedData[i])      

        for i in range(0,len(arffLoadedData)):            
            className = arffLoadedData[i][-1]
            if className not in classesHashMap:
                classesHashMap[className]=[]
            classesHashMap[className].append(arffLoadedData[i])
                       
        for i in range(0,len(arffLoadedData)):           
            className = arffLoadedData[i][-1]
            if className not in classesHashMap:
                continue
            if len(classesHashMap[className])<=2000:
                del classesHashMap[className]
                continue    
            outputDataset.append(arffLoadedData[i])
        #self.writeClassHashMap(classesHashMap)
        upSampleResults = self.upSampleClasses(outputDataset,classesHashMap) 
        classesHashMap = upSampleResults[1]  
        splitDatasetLabels = self.removeLabels(upSampleResults[0])
        return {'database':splitDatasetLabels['dataset'],'labels':splitDatasetLabels['labels'],'classesHashMap':classesHashMap}

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
        print('Will load the datasets...')
        for dataset in self.dataSetsName:
            data = arff.loadarff(settingsJson[dataset])
            totalData.extend(data[0])
        print('Raw Datasets loaded and jointed: COMPLETE')
        self.joinUDPColumns(totalData)
        print('Agregated UDP Columns: COMPLETE')

        filteredData = self.filterClasses(totalData)
        print('Upscaled the database: COMPLETE') 

        #self.normalizeData(filteredData['database'])
        #print('Normalized the database(all values in the range between 0 and 1): Complete')
                
        print('Qt. registers on database:'+str(len(filteredData['database'])))
                
        #self.writeUpSampledDatabase(filteredData['database'])
        return filteredData
   
    def loadDataset(self, preProcessed):            
        self.tensorDatabase = torch.tensor(preProcessed['database'])        
        self.tensorDatabase = torch.unsqueeze(self.tensorDatabase,dim=0)
        self.tensorDatabase = torch.unsqueeze(self.tensorDatabase,dim=0)
        tensorSizes = {}
        for tensor in self.tensorDatabase[0][0]:
            tensorSizes[len(tensor)]=0
        
        self.labelsStr=preProcessed['labels']
        self.labelsHashMap = {}
        for i in range(0,len(self.labelsStr)):
            if self.labelsStr[i] not in self.labelsHashMap:
                self.labelsHashMap[self.labelsStr[i]]=len(self.labelsHashMap.keys())                 
        self.labels = []
        for lb in self.labelsStr:
            self.labels.append(self.labelsHashMap[lb])
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):        
        return self.tensorDatabase[0][0][idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __calculate_std_mean__(self):
        return torch.std_mean(self.tensorDatabase[0][0], dim=0, keepdim=True)


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

if __name__=='__main__':    
    datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
    tonetDataset = TonetDataSet(datasets)
    preProcessed = tonetDataset.preProcessDataset()
    tonetDataset.loadDataset(preProcessed)
    train_dataloader = DataLoader(tonetDataset, batch_size=1000)
    train_features, train_labels = next(iter(train_dataloader))
    for (X, y) in enumerate(train_dataloader):
        print(X)
        print(y)