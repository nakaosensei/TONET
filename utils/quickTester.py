from datasetLoaders.datasetLoader import *

if __name__=='__main__':    
    datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
    tonetDataset = TonetDataSet(datasets)
    preProcessed = tonetDataset.preProcessDataset()
    tonetDataset.loadDataset(preProcessed)
    train_dataloader = DataLoader(tonetDataset, batch_size=1000)
    #train_features, train_labels = next(iter(train_dataloader))
    packagesPerClass = {}
    
    for (X, y) in enumerate(train_dataloader):
        
        data = y[0]
        labels = y[1]
        for i in range(0,len(labels)):
            if labels[i].item() not in packagesPerClass:
                packagesPerClass[labels[i].item()]=[]
            if labels[i].item()==5:
                packagesPerClass[labels[i].item()].extend(data[i])
            #packagesPerClass[labels[i].item()].extend(data[i])
    for k in packagesPerClass.keys():
        print(k)
        print(type(k))
        #for j in range(0, len(packagesPerClass[k])):
        #    print(packagesPerClass[k][j])
        print(len(packagesPerClass[k]))
        print(type(packagesPerClass[k]))
      
    

    
    
    
    
    
