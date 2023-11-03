import pandas as pd
import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  
    return device

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

def save3DTensorAsStringFile(tensor,filename):
    out = ""
    argMaxes = ""
    for i in range(0, len(tensor)):
        argMaxes+=str(float(tensor[i].argmax()))+'\n'
        for j in range(0,len(tensor[i])):
            if j==0:
                out+='['
            out+=str(float(tensor[i][j]))
            if j!=len(tensor[i])-1:
                out+=','
            else:
                out+=']'            
        out+='\n'
    print(filename)
    f = open(filename+'.txt','w')
    f.write(out)
    f.close()
    f = open('argmaxes.txt','w')
    f.write(argMaxes)
    f.close()

def save2DTensorAsStringFile(tensor,filename):
    out = ""
    for i in range(0, len(tensor)):
       if i==0:
           out+='['
       out+=str(float(tensor[i]))
       if i!=len(tensor)-1:
           out+=','
       else:
           out+=']'            
    out+='\n'
    print(filename)

    f = open(filename+'.txt','w')
    f.write(out)
    f.close()

