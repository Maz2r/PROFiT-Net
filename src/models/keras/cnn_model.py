from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, MaxPooling1D, Flatten, Dense

class CNNModel:
    @staticmethod
    def build_model(input_shape):
        model = Sequential()
        model.add(Conv1D(21, kernel_size=4, activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
        model.add(AveragePooling1D(pool_size=6))
        model.add(Dropout(0.01))
        model.add(Conv1D(11, kernel_size=9, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.01))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(9, kernel_size=14, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.02))
        model.add(Conv1D(9, kernel_size=3, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.01))
        model.add(Conv1D(9, kernel_size=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.01))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, kernel_initializer='he_uniform'))
        return model
