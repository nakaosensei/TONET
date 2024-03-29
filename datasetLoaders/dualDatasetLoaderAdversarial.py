from scipy.io import arff
import numpy as np
import json 
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from utils.pyTorchUtils import *
from neuralNetwork.TONetModel import *
from neuralNetwork.AttackerModel import *
import random

f = open('settings.json')
settingsJson = json.load(f)
labelsColumn = settingsJson['labelsColumn']


class DualAdversarialDataSet():

    def __init__(self, originalDatasetPath, originalLabelsPath,adversarialDatasetPath,adversarialLabelsPath, labelsPercentage):
        self.labels = None
        self.labelsStr = None
        self.labelsHashMap = {}
        self.tensorDatabase = None
        self.originalDatasetPath = originalDatasetPath
        self.originalLabelsPath = originalLabelsPath
        self.adversarialDatasetPath=adversarialDatasetPath
        self.adversarialLabelsPath=adversarialLabelsPath
        self.labelsPercentage = labelsPercentage        

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

    def loadDatasetToArray(self,datasetNm,arrayToFill,path):
        if '.pt' not in datasetNm or '.txt' in datasetNm:
            return
        tensor = torch.load(path+datasetNm)
        for register in tensor:                
            arrayToFill.append(register)


    def loadAdversarialFiles(self,totalData,labels,numberOfConsideredFiles):
        advDatasetsName = os.listdir(self.adversarialDatasetPath)  
        advDatasetsName.sort()      
        for i in range(0,len(advDatasetsName)):
            if i>numberOfConsideredFiles:
                break            
            self.loadDatasetToArray(advDatasetsName[i],totalData,self.adversarialDatasetPath)
                
        advClassesNames = os.listdir(self.adversarialLabelsPath)
        advClassesNames.sort()
        for i in range(0,len(advClassesNames)):
            if i>numberOfConsideredFiles:
                break
            self.loadDatasetToArray(advClassesNames[i],labels,self.adversarialLabelsPath)

    def convertToHashMap(self,array):
        outHash = {}
        for i in range(0,len(array)):
            if array[i] not in outHash:
                outHash[array[i].item()]=[i]
            else:
                outHash[array[i].item()].append(i)
        return outHash


    def preProcessDataset(self):
        totalData = []
        labels = []
        print('Will load the datasets...')
        dataSetsName = os.listdir(self.originalDatasetPath)  
        dataSetsName.sort()      
        for datasetNm in dataSetsName:
            self.loadDatasetToArray(datasetNm,totalData,self.originalDatasetPath)
                
        classesNames = os.listdir(self.originalLabelsPath)
        classesNames.sort()
        for datasetNm in classesNames:
            self.loadDatasetToArray(datasetNm,labels,self.originalLabelsPath)           
        numberOfSamples = len(totalData)
               
        
        numberOfConsideredFiles = int(len(dataSetsName)*(self.labelsPercentage/100))
        adversarialData = []
        adversarialLabels = []
        self.loadAdversarialFiles(adversarialData,adversarialLabels,numberOfConsideredFiles)

        numberOfInstances = numberOfConsideredFiles * 1000
        advLabelsHash = self.convertToHashMap(adversarialLabels)
        print('Classes consideradas')
        print(advLabelsHash.keys())
        intancesPerLabel = numberOfInstances/len(advLabelsHash.keys())
        print('Qt. amostras reais:'+str(numberOfSamples))
        print('Qt. amostras adversárias:'+str(numberOfInstances)+','+' qt. amostras adversárias por classe:'+str(intancesPerLabel))
        
        for lb in advLabelsHash.keys():
            consumed = 0
            while consumed < intancesPerLabel:
                for j in range(0, len(advLabelsHash[lb])):
                    if consumed>intancesPerLabel:
                        break
                    totalData.append(adversarialData[advLabelsHash[lb][j]])
                    labels.append(adversarialLabels[advLabelsHash[lb][j]])
                    consumed+=1              
        print('Raw Datasets loaded: COMPLETE')        
        filteredData = self.filterClasses(totalData, labels)
        print('Filter classes: COMPLETE')
        
        totalInstances = len(filteredData['database'])
        porcentagemFinalReais = numberOfSamples*100/totalInstances
        porcentagemFinalAdversarios = numberOfInstances*100/totalInstances
        print('Porcentagem de amostras adversarias utilizado:'+str(self.labelsPercentage)+'%')
        print('Porcentagem de amostras reais utilizado: 100%')
        print('Divisão da base após mescla:'+str(porcentagemFinalReais)+'%'+' dados reais, '+str(porcentagemFinalAdversarios)+'%'+' dados sintéticos')
        print('Qt. de registros na base:'+str(totalInstances))
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
    originalDatasetPath = '../outputs/adversarialExamples/'
    originalLabelsPath = '../outputs/targets/'
    adversarialDatasetPath = '../outputs/adversarialExamples/'
    adversarialLabelsPath = '../outputs/targets/'
    labelsPercentage = 10
    advDataset = DualAdversarialDataSet(originalDatasetPath, originalLabelsPath,adversarialDatasetPath,adversarialLabelsPath, labelsPercentage)
    preProcessed = advDataset.preProcessDataset()
    advDataset.loadDataset(preProcessed)
    train_dataloader = DataLoader(advDataset, batch_size=1000)
    train_features, train_labels = next(iter(train_dataloader))
    
    for (X, y) in enumerate(train_dataloader):
        print(X)
        print(y)
        exit()