from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(self, seq_length=4, num_outputs=15, input_shape=(80, 80)):
        img_seq_shape = (seq_length,) + input_shape + (1,)

        model = Sequential()
        model.add(Input(shape=img_seq_shape, name='img_in')) # 4*80*80*1
        model.add(TD(Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation='relu'))) # can try conv3d instead of td(conv2d)
        model.add(TD(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu')))
        model.add(TD(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu')))
        
        model.add(TD(GlobalAveragePooling2D()))

        model.add(LSTM(128, return_sequences=False, name="LSTM_fin"))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_outputs, activation='linear', name='model_outputs'))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)

        return model