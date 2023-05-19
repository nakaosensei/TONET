import torch
import cwLibrary.cw
import json
from datasetLoaders.datasetLoader import TonetDataSet
from torch.utils.data import DataLoader
f = open('settings.json')
settingsJson = json.load(f)

def run(mean,std,dataloader):
    inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),max((1 - m) / s for m, s in zip(mean, std)))

    
    # an untargeted adversary
    adversary = cw.L2Adversary(targeted=False,
                            confidence=0.0,
                            search_steps=10,
                            box=inputs_box,
                            optimizer_lr=5e-4)

    inputs, targets = next(iter(dataloader))
    adversarial_examples = adversary(net, inputs, targets, to_numpy=False)
    assert isinstance(adversarial_examples, torch.FloatTensor)
    assert adversarial_examples.size() == inputs.size()

    # a targeted adversary
    adversary = cw.L2Adversary(targeted=True,
                            confidence=0.0,
                            search_steps=10,
                            box=inputs_box,
                            optimizer_lr=5e-4)

    inputs, _ = next(iter(dataloader))
    # a batch of any attack targets
    attack_targets = torch.ones(inputs.size(0)) * 3
    adversarial_examples = adversary(net, inputs, attack_targets, to_numpy=False)
    assert isinstance(adversarial_examples, torch.FloatTensor)
    assert adversarial_examples.size() == inputs.size()


if __name__=='__main__':
    articleBatchSize = settingsJson['batchSize']
    articleEpochs = settingsJson['epochs']
    
    tonetDataset = TonetDataSet('entry12')    
    train_dataloader = DataLoader(tonetDataset, batch_size=articleBatchSize)
    
    run(0,0,train_dataloader)
    print("Done!")  
     
    