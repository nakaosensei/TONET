# Esse projeto é uma implementação particular do artigo TONet: A Fast and Efficient Method for Traffic Obfuscation Using Adversarial Machine Learning
Esta página apresenta um código que implementa os mecanismos propostos no artigo 'TONet: A Fast and Efficient Method for Traffic Obfuscation Using Adversarial Machine Learning', publicado em novembro de 2022.

O artigo propõe uma abordagem fazendo uso redes neurais adversariais visando ofuscar o tráfego de dispositivos conectados na Internet, os dados ofuscados pela solução são provados em redes neurais artificiais com altas taxas de sucesso na identificação de dispostivos.

Esse projeto contém o código fonte para a reprodução do trabalho, para o executar é preciso fazer o download das bases de dados e arquivos adicionais.

# Download das bases de dados e estrutura adicional
Faça o download do arquivo disponível em:
https://drive.google.com/file/d/1o0aA9w9pNsOWlss5IAfjcX046P3JJhUS/view?usp=sharing

Ao extrair os arquivos zipados, você verá a seguinte estrutura de pastas:
- 📂 data (Contém as bases de dados carregadas)
- 📂 inputs (Arquivos com informações sobre características da base de dados)
- 📂 outputs (Arquivos gerados, exemplos adversariais ficam aqui)
- 📂 inst-bibliotecas (Dependências do projeto)
- 📂 savedModels (Redes neurais salvas)
- 📂 src (Diretório de código do projeto, faça o git clone dentro do diretório src)

Uma vez que os arquivos estiverem descompactados em sua máquina, acesse o diretório src e executa o comando:
git clone https://github.com/nakaosensei/TONET

Alternativamente, você pode baixar um zip completo (com o diretório src já incluso) em:
https://drive.google.com/file/d/1vYwJ6hfaFKVsanDoa7pRFabRr2JPphxI/view?usp=drive_link


# Guia rápido de instalação de dependencias
Para instalação das bibliotecas necessárias, acesse a pasta **inst-bibliotecas** e execute o comando:

Comando: sudo ./downloadLibraries.sh

# Guia de execução
O primeiro passo importante é gerar a rede neural que será usada na criação das amostras adversarias, para isso, invoque o script:
```bash
python3 neuralNetworkGenerator.py
```
Ele irá treinar a rede neural que será usada para gerar os exemplos adversáriais e salvar o resultado em ../savedModels/trainedTonet.

Após isso, precisaremos treinar um possível atacante com base nos dados originais, deixamos de exemplo uma rede neural de três camadas, para treinar esse modelo atacante disponibilizado, invoque:
```bash
python3 python3 attackerNN.py
```
Ele irá treinar a rede neural que tentará realizar a classificação dos dados e salvar o resultado em ../savedModels/attackerModelWeka.

Agora basta gerar exemplos adversariais através da rede artificial gerada, utilize o script:
```bash
python3 adversarialExamplesGenerator.py
```
Na prática, o script adversarialExamplesGenerator.py utiliza a rede neural pré treinada (que está no diretório ../savedModels/trainedTonet) para gerar as asmostras adversariais e salvar em ../../outputs/adversarialExamples e ../../outputs/targets


Finalmente, para realizar o teste das amostras, use o seguinte script:
```bash
python3 adversarialExamplesTest.py
```

O script adversarialExamplesTest é quem de fato realiza o teste das amostras adversáriais contra um potencial classificador atacante, o script automaticamente invocará a rede neural atacante salva em (../savedModels/attackerModelWeka) com as amostras geradas, para isso, ele realiza testes com:
- Dados reais somente 
- Dados reais + 5% da base amostras adversárias
- Dados reais + 10% da base amostras adversárias
- Dados reais + 15% da base amostras adversárias
- Dados reais + 25% da base amostras adversárias
- Dados reais + 50% da base amostras adversárias
- Dados reais + 75% da base amostras adversárias
- Dados reais + 90% da base amostras adversárias


Para gerar os exemplos adversariais de maneira estocástica, use:
```bash
python3 stochasticAdversarialGenerator.py
```
O script stochasticAdversarialGenerator.py tenta gerar exemplos adversariais de maneira intuitiva, de modo a gerar pequenas oscilações pelo produto das grandesas dos dados originais por constantes pré-definidas, as amostras adversariais estocásticas são salvas nos diretórios ../../outputs/stochasticAdversarialExamples e ../../outputs/stochasticTargets

No momento, o teste das amostras estocásticas está fazendo a verificação considerando somente a rede neural treinada (../savedModels/trainedTonet) e os exemplos adversariais gerados como teste. Para testar as amostras geradas, use:
```bash
python3 stochasticAdversarialTester.py
```

Como bônus, foram realizados testes com outras configurações de redes neurais e do classificar k-NN sobre os dados originais, para os invocar, use:
```bash

python3 knnTester.py
```



