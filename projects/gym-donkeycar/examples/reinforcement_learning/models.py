from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
import numpy as np

## DDQN_LSTM_edit1
def get_build_model_fn(model_name):
    "return functions to build_model(), to properly format_input() state"
    ##### Baseline CNN model
    if model_name == "baseline":
        # print("baseline chosen")
        return build_model_cnnbaseline

    elif model_name == "cnnedit1":
        return build_model_cnnedit1

    elif model_name == "cnnedit1_lstm":
        return build_model_cnnedit1_lstm

    elif model_name == "cnnedit2":
        return build_model_cnnedit2
    
    elif model_name == "transfer_mobilenetv2":
        return build_model_transfer_mobilenetv2

def get_initiate_state_fn(model_name):
    if model_name in ["cnnedit1_lstm"]:
        return initiate_state_lstm

    elif model_name in ["cnnedit2"]:
        return initiate_state_cnn_stackvertical

    elif model_name in ["transfer_mobilenetv2"]:
        return initiate_state_transfer_mobilenetv2
    
    else:
        return initiate_state_cnn

def get_update_state_fn(model_name):
    if model_name in ["cnnedit1_lstm"]:
        return update_state_lstm

    elif model_name in ["cnnedit2"]:
        return update_state_cnn_stackvertical

    elif model_name in ["transfer_mobilenetv2"]:
        return update_state_transfer_mobilenetv2

    else:
        return update_state_cnn



################# initiate_state functions #################
def initiate_state_lstm(self, x_t):
    img_channels = self.img_channels      

    s_t = np.stack((x_t,)*img_channels, axis=0) # 4*80*80
    # In Keras, need to reshape 
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2], 1)  # 1*4*80*80*1
    return s_t

def initiate_state_cnn(self, x_t):
    img_channels = self.img_channels        

    s_t = np.stack((x_t,)*img_channels, axis=2) # 80*80*4
    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
    # print(s_t.shape) # debug
    return s_t

def initiate_state_cnn_stackvertical(self, x_t):
    img_channels = self.img_channels 
    color_channels = self.color_channels       

    s_t = np.vstack((x_t,)*img_channels) # 320*80
    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], color_channels)  # 1*320*80*(num_color)
    # print(s_t.shape) # debug
    return s_t

def initiate_state_transfer_mobilenetv2(self, x_t):
    if len(x_t.shape) < 3:
        x_t = np.stack((x_t, )*3, axis=2)
    img_channels = self.img_channels
    x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], x_t.shape[2])
    x_t = mobilenet_v2.preprocess_input(x_t)
    x_t = self.feature_extractor(x_t)

    s_t = np.concatenate((x_t,) * img_channels, axis=3)
    # print(s_t.shape) # debug
    return s_t

################# update_state functions #################
def update_state_cnn(self, s_t, x_t1):
    img_channels = self.img_channels

    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
    s_t1 = np.append(x_t1, s_t[:, :, :, :(img_channels-1)], axis=3)  # 1x80x80x4
    # print(s_t1.shape) # debug
    return s_t1

def update_state_lstm(self, s_t, x_t1):
    img_channels = self.img_channels
    color_channels = self.color_channels       

    x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1], color_channels)  # 1x1x80x80x(num_color)
    s_t1 = np.append(x_t1, s_t[:, :(img_channels-1), :, :], axis=1)  # 1x4x80x80x(num_color)
    return s_t1

def update_state_cnn_stackvertical(self, s_t, x_t1):
    img_rows = self.img_rows
    try:
        color_channels = x_t1.shape[2]
    except: 
        color_channels = self.color_channels       

    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], color_channels)  # 1x80x80x(num_color)
    s_t1 = np.append(x_t1, s_t[:, img_rows:, :, :], axis=1)  # 1x320x80x(num_color)
    # print(s_t1.shape) # debug
    return s_t1

def update_state_transfer_mobilenetv2(self, s_t, x_t1):
    img_channels = self.img_channels

    if len(x_t1.shape) < 3:
        x_t1 = np.stack((x_t1, )*3, axis=2)
    
    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], x_t1.shape[2])
    x_t1 = mobilenet_v2.preprocess_input(x_t1)
    x_t1 = self.feature_extractor(x_t1) # (1, 3, 3, 1280)

    s_t1 = np.append(x_t1, s_t[:, :, :, :((img_channels-1)*(x_t1.shape[3]))], axis=3) # (1, 3, 3, 5120)  
    # print(s_t1.shape) # debug
    return s_t1
################# build_model functions #################
def build_model_cnnbaseline(self):
    """
        The baseline CNN model. Note that self.state_size is (img_rows, img_cols_img_channels)
    """
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same", input_shape=self.state_size, activation='relu'))  # 80*80*4
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))

    # 15 categorical bins for Steering angles
    model.add(Dense(self.action_size, activation="linear"))

    adam = Adam(lr=self.learning_rate)
    model.compile(loss="mse", optimizer=adam) 

    return model    

def build_model_cnnedit1(self):
    """
        The CNN model with some edits 
    """
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same", input_shape=self.state_size, activation='relu'))  # 80*80*4
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    
    model.add(GlobalAveragePooling2D()) # replace the CNN model with a custom one
    
    model.add(Dense(128, activation='relu'))

    # 15 categorical bins for Steering angles
    model.add(Dense(self.action_size, activation="linear"))

    adam = Adam(lr=self.learning_rate)
    model.compile(loss="mse", optimizer=adam)

    return model

def build_model_cnnedit1_lstm(self):
    seq_length = self.img_channels
    input_shape = (self.img_rows, self.img_cols)

    img_seq_shape = (seq_length,) + input_shape + (1,)

    model = Sequential()
    model.add(Input(shape=img_seq_shape, name='img_in')) # 4*80*80*1
    model.add(TD(Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation='relu')))
    model.add(TD(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu')))
    model.add(TD(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu')))
    model.add(TD(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu')))
    model.add(TD(Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu')))
    
    model.add(TD(GlobalAveragePooling2D()))

    model.add(LSTM(128, return_sequences=False, name="LSTM_fin"))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(self.action_size, activation='linear', name='model_outputs'))

    adam = Adam(lr=self.learning_rate)
    model.compile(loss="mse", optimizer=adam)

    return model

def build_model_cnnedit2(self):
    """
        The CNN same model as edit1, except that it expects the image frames to be stacked vertically instead of in the color channel 
    """
    img_channels = self.img_channels
    img_rows = self.img_rows
    img_cols = self.img_cols
    color_channels = self.color_channels 

    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same", input_shape=(img_rows*img_channels, img_cols, color_channels), activation='relu')) # 320*80*4
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    
    model.add(GlobalAveragePooling2D()) # replace the CNN model with a custom one
    
    model.add(Dense(128, activation='relu'))

    # 15 categorical bins for Steering angles
    model.add(Dense(self.action_size, activation="linear"))

    adam = Adam(lr=self.learning_rate)
    model.compile(loss="mse", optimizer=adam)
    return model

def build_model_transfer_mobilenetv2(self):
    # feature extractor
    self.feature_extractor = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
    if self.img_cols != 96:
        self.img_cols = 96
    if self.img_rows != 96:
        self.img_rows = 96

    self.feature_extractor.train = False # use this just to extract features. no training 

    img_channels = self.img_channels

    model = Sequential()
    model.add(Input(shape=(3, 3, 1280*img_channels), name='features_in')) # this one is from the output of the mobilenet model
    model.add(GlobalAveragePooling2D()) 

    model.add(Dense(128, activation="relu"))
    
    # 15 categorical bins for Steering angles
    model.add(Dense(15, activation="linear"))

    adam = Adam(lr=self.learning_rate)
    model.compile(loss="mse", optimizer=adam)

    return model