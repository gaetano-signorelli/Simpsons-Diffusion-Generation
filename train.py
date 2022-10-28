import os

from src.model.callbacks import SaveUpdateStepCallback
from src.model.model_handler import ModelHandler

from src.model.config import *

if __name__ == '__main__':

    input_shape = (IMAGE_RESIZE[0],IMAGE_RESIZE[1],3)

    model_handler = ModelHandler(input_shape, N_HEADS, TIME_EMBEDDING_SIZE,
                                BETA_START, BETA_END, NOISE_STEPS,
                                load_model=True)
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
                shuffle=False,
                seed=None,
                validation_split=None,
                subset=None,
                interpolation="bilinear",
                follow_links=True,
                crop_to_aspect_ratio=False,
            )

    model_handler.model.fit(generator,
                            callbacks=[callback],
                            epochs=remaining_epochs)
