import os
import argparse
from PIL import Image
import numpy as np
from keras.models import load_model

def segment_image(image_path, model_path, output_path):
    model = load_model(model_path, compile=False)
    img = Image.open(image_path).convert('RGB')
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    prediction = model.predict(img_array)
    prediction_image = Image.fromarray((prediction.squeeze() * 255).astype(np.uint8))
    prediction_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Realiza a inferência de imagens usando um modelo U-Net.")
    parser.add_argument('--rgb', type=str, required=True, help='Caminho para a imagem RGB a ser segmentada.')
    parser.add_argument('--modelpath', type=str, required=True, help='Caminho para o modelo treinado.')
    parser.add_argument('--output', type=str, required=True, help='Caminho para salvar a imagem segmentada.')

    args = parser.parse_args()

    segment_image(args.rgb, args.modelpath, args.output)

if __name__ == "__main__":
    main()

#python3 src/inferencia/inference_model.py --rgb dados/blocos/Orthomosaico_teste/bloco_0_0.png --modelpath dados/modelo/unet_model.h5 --output dados/inferencia/teste.png
