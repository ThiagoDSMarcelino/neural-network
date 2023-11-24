from keras import optimizers
from keras import callbacks
from keras import metrics
from keras import losses
from keras.preprocessing import image
from calc import calc_steps_per_epoch, calc_validation_steps
from ai import gen_model
import shutil
from images_util import delete_corrupted_images

def delete_dir(directory: str):
    try:
        shutil.rmtree(directory)
        print(f"Directory '{directory}' successfully removed.")
    except OSError as e:
        print(f"Error: {e}")

train_size = 0.8
batch_size = 32
data_path = 'data'
save_path = 'epochs/'

delete_corrupted_images(data_path)
delete_dir(save_path)

model = gen_model()

# Building
model.compile(
    optimizer=optimizers.SGD(),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.CategoricalAccuracy()]
)

data_gen = image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=1 - train_size
)

X_train = data_gen.flow_from_directory(
    data_path,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

X_test = data_gen.flow_from_directory(
    data_path,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

steps_per_epoch = calc_steps_per_epoch(batch_size, train_size)
validation_steps = calc_validation_steps(steps_per_epoch)

model.fit(
    x=X_train,
    validation_data=X_test,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_steps=validation_steps,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='loss',
            patience=4
        ),
        callbacks.ModelCheckpoint(
            filepath=save_path +'model_{epoch:02d}_{loss:.2f}.keras'
        )
    ]
)

model.save('model')