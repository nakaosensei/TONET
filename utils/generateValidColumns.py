from scipy.io import arff
import json 
import pandas as pd
f = open('settings.json')
settingsJson = json.load(f)

atributosSelecionados = [160,161,162,164,165,181,182,183,185,186,195,196,197,199,200,202,203,204,206,207]

data = arff.loadarff(settingsJson['entry12'])
df = pd.DataFrame(data[0])
mapaColunas = []
colunasNaoConsideradas=[]
colunasConsideradas=[]
i = 1
for c in df.columns:
    if i not in atributosSelecionados:
        colunasNaoConsideradas.append(c)
    else:
        colunasConsideradas.append(c)
    mapaColunas.append(str(i)+'----'+str(c))
    i+=1

settingsJson['columnMapping']=mapaColunas
settingsJson['colunasConsideradas']=colunasConsideradas
settingsJson['colunasNaoConsideradas']=colunasNaoConsideradas
with open('settings.json', 'w') as f:
    json.dump(settingsJson, f)