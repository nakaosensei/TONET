from datasetLoaders.datasetLoader import TonetDataSet
from torch.utils.data import DataLoader
from utils.pyTorchUtils import save3DTensorAsStringFile,save2DTensorAsStringFile
import torch

featuresPath='../outputs/stochasticAdversarialExamples/' #Caminho dos arrays com exemplos adversariais
labelsPath='../outputs/stochasticTargets/' #Caminho das classes respectivas
'''
Exemplo de arquivo de caracteristicas:
Cada instância (linha) de um arquivo de exemplos adversariais contém um array com 20 elementos, que são as 20 cartacterísticas
associadas a tamanho de pacote e aos intervalos de chegada e saida dos pacotes utilizadas no trabalho TONet
[40.0,46.0,81.0,576.0,576.0,40.0,40.0,40.0,48.0,132.0,0.0,7.300000288523734e-05,0.0007800000021234155,0.04965199902653694,1.7886840105056763,0.0,7.300000288523734e-05,0.0007800000021234155,0.04965199902653694,1.7886840105056763]
[40.0,576.0,576.0,576.0,576.0,40.0,40.0,40.0,40.0,132.0,9.999999974752427e-07,5.400000009103678e-05,7.79999973019585e-05,0.0002460000105202198,2.881584882736206,9.999999974752427e-07,5.400000009103678e-05,7.79999973019585e-05,0.0002460000105202198,2.881584882736206]
[40.0,576.0,576.0,576.0,576.0,40.0,40.0,40.0,40.0,132.0,9.999999974752427e-07,5.500000042957254e-05,0.0001340000017080456,0.0008779999916441739,1.6371829509735107,9.999999974752427e-07,5.500000042957254e-05,0.0001340000017080456,0.0008779999916441739,1.6371829509735107]
[40.0,46.0,81.0,576.0,576.0,40.0,40.0,40.0,48.0,132.0,1.5999999959603883e-05,9.300000237999484e-05,0.001184999942779541,0.04963900148868561,1.6525559425354004,1.5999999959603883e-05,9.300000237999484e-05,0.001184999942779541,0.04963900148868561,1.6525559425354004]
[40.0,46.5,188.0,576.0,576.0,40.0,40.0,40.0,48.0,132.0,9.999999974752427e-07,6.600000051548705e-05,0.0003279999946244061,0.049681998789310455,1.530735969543457,9.999999974752427e-07,6.600000051548705e-05,0.0003279999946244061,0.049681998789310455,1.530735969543457]
[40.0,576.0,576.0,576.0,576.0,40.0,40.0,40.0,40.0,132.0,9.999999974752427e-07,5.500000042957254e-05,0.00011700000322889537,0.001398000051267445,1.7441760301589966,9.999999974752427e-07,5.500000042957254e-05,0.00011700000322889537,0.001398000051267445,1.7441760301589966]
[40.0,576.0,576.0,576.0,576.0,40.0,40.0,40.0,40.0,132.0,9.999999974752427e-07,5.400000009103678e-05,8.299999899463728e-05,0.0002579999854788184,1.6271990537643433,9.999999974752427e-07,5.400000009103678e-05,8.299999899463728e-05,0.0002579999854788184,1.6271990537643433]

Exemplo de arquivo de labels:
É um arquivo que contém as classes respectivas as instâncias do arquivo de característica, no artigo são consideradas 6 classes
distintas, no exemplo abaixo temos um array em que cada posição é referente a classe das instâncias do exemplo de arquivo de características
[0,1,2,1,0,4,0]

As bases de dados cruas utilizadas estão salva no diretório ../data/discriminationsFlowBasedClassification, no formato .arff,
mas voce nao precisa se preocupar em usar elas, pois ela ja foi filtrada e transformada de acordo com as especificacoes do 
artigo, os dados originais estao salvos nos diretorios ../outputs/originalDatabaseSamples e ../outputs/originalDatabaseTargets.

Felizmente ja existem codigos que lidam com o carregamento e pre-processamento dos dados, observe que no codigo main abaixo,
o dataset processado esta sendo carregado, e depois percorrido. 
Sugestao: Experimente executar o codigo e printar as variveis data e labels.

Nessa tarefa, voce tem dois objetivos iniciais:
1) Gerar amostras adversariais
2) Averiguar como a rede neural TONet pre-treinada do artigo se comporta quando for alimentada com as suas amostras

Para gerar as amostras adversariais, sera necessario voce observar a natureza das caracteristicas para pensar em
que tipo de transformacao seria aplicavel.
Mas pra algo bem simples, pra testar num primeiro momento, todas as 20 caracteristicas de cada instancia 
sao atributos numericos, entao voce pode experimentar aumentar ou diminuir os valores desses atributos em 5%, por exemplo

Bastaria voce modificar os valores da variavel data, dentro do for loop do codigo abaixo, no codigo ja esta o comando para 
salvar os arquivos no formato .txt e .pt, que sao salvos nas pastas ../outputs/stochasticAdversarialExamples/ e 
../outputs/stochasticTargets/  

Depois de gerar as amostras adversariais, va ao arquivo stochasticAdversarialTester para 
testar as amostras em uma rede neural pre-treinada com os dados originais do dataset.
'''

if __name__=='__main__':    
    datasets = ['entry1','entry2','entry3','entry4','entry5','entry6','entry7','entry8','entry9','entry10']
    tonetDataset = TonetDataSet(datasets)
    preProcessed = tonetDataset.preProcessDataset()
    tonetDataset.loadDataset(preProcessed)
    dataloader = DataLoader(tonetDataset, batch_size=1000)
    i=0
    for (X, y) in enumerate(dataloader):
        data = y[0]
        labels = y[1]
        

        '''
        for i in range(0, len(data)):
            for j in range(0, len(instancia)):
                data[i][j] = data[i][j]*0.98
        '''

        save3DTensorAsStringFile(data,featuresPath+str(i))
        save2DTensorAsStringFile(labels,labelsPath+str(i))
        torch.save(data, featuresPath+'original'+str(i)+'.pt')
        torch.save(labels, labelsPath+'original'+str(i)+'.pt')
        i+=1
        