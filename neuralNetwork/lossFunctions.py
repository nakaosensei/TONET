import torch
lambdaDistance2Norm = 4000 
lambdaConstraint2Norm = 100
lambdaDistanceInfiniteNorm = 5000
lambdaConstraintInfiniteNorm = 10


def contraintLoss2Norm(my_outputs, my_labels):
    maxVariance = torch.max(my_outputs)
    result = 1/len(my_outputs)*sumOutputs
    result*=lambdaDistance2Norm

def distanceLoss2Norm(my_outputs, my_labels):
    sumOutputs = torch.sum(my_outputs)
    result = 1/len(my_outputs)*sumOutputs
    result*=lambdaDistance2Norm
    '''print('outputs')
    print(my_outputs)
    print('labels')
    print(my_labels)
    print(len(my_outputs))
    print(len(my_outputs[0]))
    '''
    return result

def distanceLossInfiniteNorm(my_outputs, my_labels):
    sumOutputs = torch.sum(my_outputs)
    result = 1/len(my_outputs)*sumOutputs
    result*=lambdaDistanceInfiniteNorm
    return result