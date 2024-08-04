from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, UpSampling2D, concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

def jaccard_loss(y_true, y_pred):
    smooth = 1e-10
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - jac

def jaccard_coef(y_true, y_pred):
    smooth = 1e-10
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def build_unet_model(input_shape, upconv=True, droprate=0.5):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(droprate)(pool4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv5)
    drop5 = Dropout(droprate)(conv5)

    if upconv:
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv4])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4])
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = Dropout(droprate)(conv6)

    if upconv:
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = Dropout(droprate)(conv7)

    if upconv:
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = Dropout(droprate)(conv8)

    if upconv:
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    return model
