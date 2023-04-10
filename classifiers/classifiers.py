from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')
'''
Metodo mapear as chaves codificadas com as decodificadas
Retorna um hashMap que para toda chave codificada(tipo inteiro), o valor seja
o seu respective endereco MAC.
'''
def mapDevices(encodedLabels, decodedLabels):
    encodedHash = {}
    for i in range(0, len(encodedLabels)):
        encodedHash[str(encodedLabels[i])]=decodedLabels[i]
    return encodedHash



'''
Recebe como parametros:
model: Classificador utilizado, nessa implementacao foram testadas instancias de
RandomForestClassifier() e KNeighborsClassifier()

labels: Labels reais dos dispositivos usadas para o conjunto de treino, se trata
de uma lista de enderecos MAC que serao usadas como labels pelo classificador


characteristics: objeto que contem os arrays de caracteristiscas, ex:
{"razaoBytes":[200,300],"bytesSend":[200,300],"bytesReceived":[200,300],"framesSend":[10,10],"framesReceived":[10,10],"rsr":[2.1,3.2],"rbf":[2.1,3.2]}


Esse script faz as seguintes tarefas:
1) Codifica as labels para valores numericos (encodedLabels)
2) Agrupa os arrays de caracteristicas na variavel features
3) Separa os conjuntos de treino e teste, 50% para cada
4) Executa o classificador passado por parametro passando os conjuntos de treino e teste
5) Percorre o hash de resultados, para cada endereco mac e carregado em uma estrutura da seguinte forma:
{"mac1":{"precision":0.96},"mac2":"precision":0.88}
6) O objeto hash montado e retornado
'''
def mountResultsTrainTest(model,labels, characteristics):
    labelEncoder = preprocessing.LabelEncoder()
    
    encodedLabels=labelEncoder.fit_transform(labels)
    decodedLabels = labelEncoder.inverse_transform(encodedLabels)
    features=characteristics

    # Train the model using the training sets
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), encodedLabels, test_size=0.50, random_state=42)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    decodedResult = labelEncoder.inverse_transform(y_pred)
    resultMetrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    map = mapDevices(encodedLabels,decodedLabels)

    outputJson = {}
    for label in resultMetrics.keys():
        if label not in map:
            continue
        outputJson[map[label]]={}
        outputJson[map[label]]['precision']=resultMetrics[label]['precision']
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return outputJson


'''
Executa o classificador randomForest para o array de caracteristicas e labels
passadas como parametro, 50 niveis sao usados por padrao.
'''
def runRandomForest(labels, characteristics):
    model = RandomForestClassifier(max_depth=50, random_state=0)
    outputJson = mountResultsTrainTest(model,labels, characteristics)
    return outputJson

'''
Executa o classificador KNN para o array de caracteristicas e labels
passadas como parametro, k=3 por padrao.
'''
def runKnn(labels, characteristics, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    outputJson = mountResultsTrainTest(model,labels, characteristics)
    return outputJson

'''
Executa o classificador AdaBoost para o array de caracteristicas e labels
passadas como parametro.
'''
def runAdaBoost(labels, characteristics, n_estimators=50):
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=5)
    outputJson = mountResultsTrainTest(model,labels, characteristics)
    return outputJson


