import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers.embeddings import Embedding
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from keras.layers import *
import keras_tuner as kt

from tensorflow import keras

from ium_recommendations.features.build_features import to_numpy_sequences

class ModelBase:
    __model = tf.keras.Model()
    max_session_length = 10
    max_products = 319
    max_categories = 15

    validation_split = 0.2
    batch_size = 64

    hp: kt.HyperParameters
    tuner: kt.Hyperband
    stop_early: None

    MODELS_PATH = "models\\"

    def __init__(self, max_session_length, max_products, max_categories) -> None:
        self.max_session_length = max_session_length
        self.max_products = max_products
        self.max_categories = max_categories
        self.hp = kt.HyperParameters()
        self.hp.values = { # default hyperparameters values
            'lstm_units': 128,
            'learning_rate': 0.001,
            'hp_embedding_vector_length': 128,
            'hp_embedding_vector_length1': 128,
            'hp_embedding_vector_length2': 128
        }
        self.tuner = kt.Hyperband(self.model_builder,
                     objective='val_accuracy',
                     max_epochs=10)
        self.stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    def model_builder(self, hp):
        raise "Empty model"
    
    def create(self):
        self.__model = self.tuner.hypermodel.build(self.hp)
        return

    def predict(self, x):
        y_predict = self.__model.predict(numpy.array([x,]))
        return numpy.argmax(y_predict) # reverse to_categorical()
    
    def predict(self, x1, x2):
        y_predict = self.__model.predict(([numpy.array([x1,]), numpy.array([x2,])]))
        return numpy.argmax(y_predict) # reverse to_categorical()

    def train(self, x_train, y_train, epochs):
        y_train = to_categorical(y_train, num_classes=self.max_products+1)
        history = self.__model.fit(x_train, y_train, epochs=epochs, validation_split=self.validation_split, batch_size=64)
        return history

    def evaluate(self, x_test, y_test):
        y_test = to_categorical(y_test, num_classes=self.max_products+1)
        scores = self.__model.evaluate(x_test, y_test, verbose=0)
        return scores

    def load(self, name):
        self.__model = keras.models.load_model(self.MODELS_PATH+name)
        return

    def save(self, name):
        self.__model.save(self.MODELS_PATH+name)
        return

    def tune_parameters(self, x_train, y_train, epochs):
        y_train = to_categorical(y_train, num_classes=self.max_products+1)
        self.tuner.search(x_train, y_train, epochs=epochs, validation_split=self.validation_split, callbacks=[self.stop_early])
        self.hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        return self.hp
    
    def summary(self):
        return self.__model.summary()

    def plot_model(self):
        tf.keras.utils.plot_model(self.__model, show_shapes=True, show_layer_names=True)

class ModelSimple(ModelBase):
    def model_builder(self, hp):
        # Hyperparameters
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_lstm_units = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])

        model = Sequential()
        model.add(Input((self.max_session_length, self.max_products+1)))
        model.add(LSTM(hp_lstm_units))
        model.add(Dense(self.max_products+1, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        return model
    
    def train(self, x_train, y_train, epochs):
        x_train = to_categorical(x_train, num_classes=self.max_products+1)
        return super().train(x_train, y_train, epochs)
    
    def evaluate(self, x_test, y_test):
        x_test = to_categorical(x_test, num_classes=self.max_products+1)
        return super().evaluate(x_test, y_test)

    def tune_parameters(self, x_train, y_train, epochs):
        x_train = to_categorical(x_train, num_classes=self.max_products+1)
        return super().tune_parameters(x_train, y_train, epochs)

    

class ModelEmbedding(ModelBase):
    def model_builder(self, hp):
        # Hyperparameters
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_embedding_vector_length = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])
        hp_lstm_units = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])

        model = Sequential()
        model.add(Input((self.max_session_length)))
        model.add(Embedding(self.max_products+1, hp_embedding_vector_length, input_length=self.max_session_length))
        model.add(LSTM(hp_lstm_units))
        model.add(Dense(self.max_products+1, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        return model

class ModelCategories(ModelBase):
    def model_builder(self, hp):
        # Hyperparameters
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_embedding_vector_length1 = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])
        hp_embedding_vector_length2 = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])
        hp_lstm_units = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])

        input1 = keras.layers.Input(shape=(self.max_session_length))
        input2 = keras.layers.Input(shape=(self.max_session_length))
        embedded1 = keras.layers.Embedding(self.max_products+1, hp_embedding_vector_length1, input_length=self.max_session_length)(input1)
        embedded2 = keras.layers.Embedding(self.max_categories+1, hp_embedding_vector_length2, input_length=self.max_session_length)(input2)
        merged = keras.layers.Concatenate(axis=1)([embedded1, embedded2])
        lstm = keras.layers.LSTM(hp_lstm_units)(merged)
        output = keras.layers.Dense(self.max_products+1, activation=keras.activations.softmax)(lstm)
        model = keras.Model(inputs=[input1, input2], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        return model