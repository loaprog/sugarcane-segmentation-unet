# Desafio-IA
O objetivo final é criar um modelo de rede neural que possa segmentar imagens para detectar vegetação, utilizando um método de binarização para distinguir solo exposto e áreas de vegetação.

![image](https://github.com/user-attachments/assets/efab4c43-2306-4141-94d8-d3144a81e3fb) ![image](https://github.com/user-attachments/assets/902b6da4-22b9-4967-ba36-703c5ac3b633) ![image](https://github.com/user-attachments/assets/88eabcc8-6bdb-485f-beb3-471250e3cc13) ![image](https://github.com/user-attachments/assets/659cf8d7-7dd2-4939-a02b-b8b0b1ab3c3e) ![image](https://github.com/user-attachments/assets/4cdfa5ef-59c1-4485-9dc2-1aa0bd6f7333) ![image](https://github.com/user-attachments/assets/69fbd204-eb2a-4ebc-bb88-effaa92faf73)

## 🤖 Arquitetura e Implementação do Modelo
O projeto utiliza a arquitetura U-Net para segmentação de imagens, uma rede neural convolucional altamente eficaz para tarefas de segmentação. O modelo é composto por um encoder que extrai características de diferentes níveis de resolução da imagem e um decoder que reconstrói a imagem segmentada a partir dessas características. A arquitetura é projetada para combinar informações de diferentes camadas através de concatenamento, preservando detalhes importantes enquanto aumenta a resolução da imagem.

<img src="https://github.com/user-attachments/assets/1fbaaa6e-323a-4372-b1d3-843bd50a205c" width="550"/>

Também utiliza o Índice de Vegetação por Diferença de Verde (GLI) para melhorar a segmentação. O GLI é um índice de vegetação que ajuda a quantificar a presença de vegetação em imagens, o que é particularmente útil para distinguir baixar biomassa e alta biomassa. O índice é calculado a partir das bandas de cores da imagem e, em seguida, a imagem resultante é binarizada para criar um mapa de vegetação e solo. Esses dados binarizados são usados como parte do treinamento do modelo U-Net para melhorar a precisão da segmentação.

<img src="https://github.com/user-attachments/assets/069bfb96-acb4-436b-abd0-a0188d0b0c01" alt="Descrição da Imagem" width="300"/>

 ## 💣💣💣 Antes de Rodar (*importante*)
 - Antes de executar o modelo, é essencial preparar os dados necessários. Primeiro, acesse o link do Google Drive para baixar a pasta contendo os dados base, que inclui ortomosaicos brutos e divididos em blocos, imagens segmentadas, modelos treinados e resultados de inferência.
   
https://drive.google.com/drive/folders/1xO1kjO2bOwpvkwe5uxx1z6Dfb-wSs9u7?usp=sharing

## 🚀 Configuração

Podemos rodar via pip ou docker:

### 1. Via pip install  

Rode o seguinte comando:

	pip install -r requirements.txt
 
### 2. Via container docker

Rode o seguinte comando:

	docker build -t <image_name> .

   	docker run -it <image_name> /bin/bash

 ## 🐛 Como Rodar

O projeto é dividido em quatro/cinco etapas principais:

1. **Quebra de Imagem em Blocos**: Divida a imagem ortomosaica em blocos usando o comando:
   ```bash
   python divide_orthomosaic.py --input </path/to/orthomosaic.tif> --output </path/to/output/dir/>
2. **Geração de Dataset**: Segmente as imagens em blocos usando o índice de vegetação GLI com o comando:
   ```bash
   python binarize_images.py --input </path/to/images/dir> --output </path/to/segmented/dir/>
3. **Implementação e Treinamento da Rede Neural U-Net**: Treine o modelo com o comando:
   ```bash
   python train_model.py (verifique x_train, y_train e output_model está certo) 
4. **Inferência do Modelo**: Realize a inferência em uma imagem específica usando:
   ```bash
   python inference_model.py --rgb </path/to/image.png> --modelpath </path/to/model.h5> --output </path/to/segmented/image.png>
4. **Para processar todas as imagens em uma pasta**, utilize:
   ```bash
   python inference_folder.py (verifique model_path, output_folder e dados_blocos está certo) 
   ```

## 👽 Estrutura e Organização do Projeto 
Este projeto está estruturado da seguinte forma para garantir uma organização clara e eficiente dos dados, código-fonte e outros arquivos importantes:

![image](https://github.com/user-attachments/assets/12526af3-d994-4672-867a-c370ffeaedca)
