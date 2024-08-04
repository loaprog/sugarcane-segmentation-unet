import os
from PIL import Image, ImageChops
import numpy as np
from keras.models import load_model
# from keras import backend as K

model_path = 'dados/modelo/unet_model.h5'
output_folder = 'dados/inferencia/Orthomosaico_teste/'
dados_blocos = 'dados/blocos/Orthomosaico_teste/'

model = load_model(model_path, compile=False)

for file_name in os.listdir(dados_blocos):
    try:
        img = Image.open(os.path.join(dados_blocos, file_name)).convert('RGB')
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        prediction_image = Image.fromarray((prediction[0].squeeze() * 255).astype(np.uint8))

        combined_image = Image.new('RGBA', (256, 128))
        combined_image.paste(img.convert('RGBA'), (0, 0))
        combined_image.paste(prediction_image.convert('RGBA'), (128, 0))

        combined_output_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_combined.png')
        combined_image.save(combined_output_path)
        print(f"Imagem combinada para {file_name} salva em {combined_output_path}")
    except Exception as e:
        print(f"Erro ao processar a imagem {file_name}: {e}")