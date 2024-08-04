import os
from PIL import Image
import argparse

def dividir_imagem(caminho_entrada, dir_saida, tamanho_bloco):
    img = Image.open(caminho_entrada)
    largura_img, altura_img = img.size
    for i in range(0, largura_img, tamanho_bloco):
        for j in range(0, altura_img, tamanho_bloco):
            caixa = (i, j, i + tamanho_bloco, j + tamanho_bloco)
            img_corte = img.crop(caixa)
            img_corte.save(os.path.join(dir_saida, f'bloco_{i}_{j}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ortomosaico em blocos.")
    parser.add_argument('--input', type=str, required=True, help='imagem do ortomosaico.')
    parser.add_argument('--output', type=str, required=True, help='diretório de saída.')
    parser.add_argument('--tamanho_bloco', type=int, default=128, help='Tamanho de cada bloco.')
    args = parser.parse_args()
    dividir_imagem(args.input, args.output, args.tamanho_bloco)


# python3 "/home/leonardo-alves/Área de Trabalho/IA/src/preparacao_dados/divide_orthomosaic.py" --input "/home/leonardo-alves/Área de Trabalho/IA/dados/brutos/Orthomosaico_teste.tif" --output "/home/leonardo-alves/Área de Trabalho/IA/dados/blocos/Orthomosaico_teste/"

