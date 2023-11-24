import os

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from keras import metrics
from keras import losses

from art_style.ai import gen_model
from util.files import delete_dir
from util.image import count_images, delete_corrupted_images

current_path = os.path.dirname(os.path.abspath(__file__))

train_path = current_path + '/data/training_set'
validation_path = current_path + '/data/validation_set'
save_path = current_path + '/epochs/'

def clean_data():
    delete_corrupted_images(train_path)
    delete_corrupted_images(validation_path)
    delete_dir(save_path)

def train_model():
    train_batch_size = 32
    validation_batch_size = 16

    steps_per_epoch = count_images(train_path) // train_batch_size
    validation_steps = count_images(validation_path) // validation_batch_size

    train_data_gen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
    )

    validation_data_gen = ImageDataGenerator(
        rescale=1.0/255,
    )

    X_train = train_data_gen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=train_batch_size,
        class_mode='categorical'
    )

    X_validation = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(64, 64),
        batch_size=validation_batch_size,
        class_mode='categorical'
    )

    model = gen_model()

    model.compile(
        optimizer=optimizers.SGD(),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.CategoricalAccuracy()]
    )

    model.fit(
        x=X_train,
        validation_data=X_validation,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_steps=validation_steps,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='loss',
                patience=4
            ),
            callbacks.ModelCheckpoint(
                filepath=save_path + 'model_{epoch:02d}_{loss:.2f}.keras'
            )
        ]
    )