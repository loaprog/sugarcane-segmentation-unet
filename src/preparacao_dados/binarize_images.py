import os
from PIL import Image
import numpy as np
import argparse

def calcular_gli(imagem_rgb):
    array_rgb = np.array(imagem_rgb)
    R = array_rgb[:, :, 0].astype(float)
    G = array_rgb[:, :, 1].astype(float)
    B = array_rgb[:, :, 2].astype(float)
    gli = (2 * G + R + B) / (2 * G - R - B)
    return gli

def binarizar_imagem(dir_entrada, dir_saida):
    limiar = 0.5  
    for nome_arquivo in os.listdir(dir_entrada):
        if nome_arquivo.endswith(".png"):
            caminho_img = os.path.join(dir_entrada, nome_arquivo)
            img_rgb = Image.open(caminho_img).convert("RGB")
            gli_img = calcular_gli(img_rgb)
            img_gli = Image.fromarray(gli_img)
            
            gli_img[gli_img > limiar] = 255
            gli_img[gli_img <= limiar] = 0
            img_binaria = Image.fromarray(gli_img.astype(np.uint8))
            img_binaria.save(os.path.join(dir_saida, nome_arquivo))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarizar usando o Índice de Vegetação GLI")
    parser.add_argument('--input', type=str, required=True, help='imagens de entrada')
    parser.add_argument('--output', type=str, required=True, help='imagens binarizadas.')
    args = parser.parse_args()
    binarizar_imagem(args.input, args.output)

# python3 "src/preparacao_dados/binarize_images.py" --input "dados/blocos/" --output "dados/segmentadas/"

