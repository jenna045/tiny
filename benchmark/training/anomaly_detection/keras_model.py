"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import tensorflow.keras as keras
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))

    h = Dense(128,name="Dense_1")(inputLayer)
    h = BatchNormalization(name="Batch_Normalization_1")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_2")(h)
    h = BatchNormalization(name="Batch_Normalization_2")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_3")(h)
    h = BatchNormalization(name="Batch_Normalization_3")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_4")(h)
    h = BatchNormalization(name="Batch_Normalization_4")(h)
    h = Activation('relu')(h)
    
    h = Dense(8,name="Dense_5")(h)
    h = BatchNormalization(name="Batch_Normalization_5")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_6")(h)
    h = BatchNormalization(name="Batch_Normalization_6")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_7")(h)
    h = BatchNormalization(name="Batch_Normalization_7")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_8")(h)
    h = BatchNormalization(name="Batch_Normalization_8")(h)
    h = Activation('relu')(h)

    h = Dense(128,name="Dense_9")(h)
    h = BatchNormalization(name="Batch_Normalization_9")(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)
    
    return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

    
