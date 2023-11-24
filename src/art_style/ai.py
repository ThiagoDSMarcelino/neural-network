from keras import models
from keras import layers
from keras import activations

def preprocessing_layers(model: models.Model) -> models.Model:
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        input_shape=(64, 64, 3),
        activation=activations.relu
    ))

    model.add(layers.MaxPool2D(
        pool_size=(2, 2)
    ))

    model.add(layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        input_shape=(30, 30, 3),
        activation=activations.relu
    ))

    model.add(layers.MaxPool2D(
        pool_size=(2, 2)
    ))

    model.add(layers.Flatten())

    return model

def train_layers(model: models.Model) -> models.Model:
    model.add(layers.Dense(
        units= 128,
        kernel_initializer='random_normal',
        bias_initializer='zeros',
        activation=activations.relu
    ))

    model.add(layers.Dense(
        units= 64,
        kernel_initializer='random_normal',
        bias_initializer='zeros',
        activation=activations.relu
    ))

    model.add(layers.Dense(
        units= 5,
        kernel_initializer='random_normal',
        bias_initializer='zeros',
        activation=activations.softmax
    ))

    return model

def gen_model() -> models.Sequential:
    model = models.Sequential()

    model = preprocessing_layers(model)

    model = train_layers(model)

    return model