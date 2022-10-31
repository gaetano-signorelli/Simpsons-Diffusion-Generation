import tensorflow as tf

from src.model.callbacks import SaveUpdateStepCallback
from src.model.model_handler import ModelHandler

from src.model.config import *

if __name__ == '__main__':

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    input_shape = (IMAGE_RESIZE[0],IMAGE_RESIZE[1],3)

    model_handler = ModelHandler(input_shape, N_HEADS, TIME_EMBEDDING_SIZE,
                                BETA_START, BETA_END, NOISE_STEPS,
                                load_model=LOAD_MODEL)
    model_handler.build_model()

    callback = SaveUpdateStepCallback(model_handler)

    remaining_epochs = EPOCHS - model_handler.current_step

    generator = tf.keras.utils.image_dataset_from_directory(
                TRAIN_PATH,
                labels=None,
                label_mode=None,
                class_names=None,
                color_mode="rgb",
                batch_size=BATCH_SIZE,
                image_size=IMAGE_RESIZE,
                shuffle=True,
                seed=None,
                validation_split=None,
                subset=None,
                interpolation="bilinear",
                follow_links=True,
                crop_to_aspect_ratio=False,
            )

    generator = generator.map(lambda x: x/255)

    model_handler.model.fit(generator,
                            callbacks=[callback],
                            epochs=remaining_epochs)#,
                            #batch_size=BATCH_SIZE)
