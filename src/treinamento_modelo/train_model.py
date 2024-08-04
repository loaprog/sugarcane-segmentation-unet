import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from unet_model import build_unet_model, jaccard_loss, jaccard_coef

x_train_path = 'dados/modelo/x_train.npy'
y_train_path = 'dados/modelo/y_train.npy'
output_model = 'dados/modelo/unet_model.h5'

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)

input_shape = x_train.shape[1:]
model = build_unet_model(input_shape)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=jaccard_loss, metrics=[jaccard_coef, 'accuracy'])
model.summary()

checkpoint = ModelCheckpoint(output_model, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

model.fit(x_train, y_train, validation_split=0.2, batch_size=8, epochs=50, callbacks=[checkpoint, early_stop])
