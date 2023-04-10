# Esse projeto é uma implementação particular do artigo TONet: A Fast and Efficient Method for Traffic Obfuscation Using Adversarial Machine Learning
Esta página apresenta um código que implementa os mecanismos propostos no artigo 'TONet: A Fast and Efficient Method for Traffic Obfuscation Using Adversarial Machine Learning', publicado em novembro de 2022.

O artigo propõe uma abordagem fazendo uso redes neurais adversariais visando ofuscar o tráfego de dispositivos conectados na Internet, os dados ofuscados pela solução são provados em redes neurais artificiais com altas taxas de sucesso na identificação de dispostivos.

Esse projeto contém o código fonte para a reprodução do trabalho, para o executar é preciso fazer o download das bases de dados e arquivos adicionais.

# Download das bases de dados e estrutura adicional
Faça o download do arquivo disponível em:
https://drive.google.com/file/d/1V-2RsMFhZZZe4eIr3aFays657RxNMngM/view?usp=sharing

Ao extrair os arquivos zipados, você verá a seguinte estrutura de pastas:
- 📂 data (Contém as bases de dados carregadas)
- 📂 inputs (Arquivos com informações sobre características da base de dados)
- 📂 inst-bibliotecas (Dependências do projeto)
- 📂 savedModels (Redes neurais salvas)
- 📂 src (Diretório de código do projeto, faça o git clone dentro do diretório src)


# Guia rápido de instalação
Para instalação das bibliotecas necessárias, acesse a pasta **inst-bibliotecas** e execute o comando:

Comando: sudo ./downloadLibraries.sh

Uma vez que os arquivos estiverem descompactados em sua máquina, acesse o diretório src e executa o comando:
git clone https://github.com/nakaosensei/TONET


# Guia de execução
Uma vez que todas as dependências forem supridas, acesse o diretório src, para gerar a rede neural proposta no trabalho TONet, faça:
```bash
python3 tonetNN.py
```

Além da rede TONet, em diversos trabalhos é citada uma rede neural atacante, com uma configuração diferente nas camadas internas, para testar essa rede:
```bash
python3 attackerNN.py
```

Por fim, existe um script que usa o classificador k-NN para a classificação dos pacotes, para o executar: 
```bash
python3 knnTester.py
```

