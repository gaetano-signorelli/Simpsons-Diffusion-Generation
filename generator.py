import os
import numpy as np
from PIL import Image
import argparse
from src.model.model_handler import ModelHandler

from src.model.config import *

def parse_arguments():

    parser = argparse.ArgumentParser(description='Diffusion sampling generation')
    parser.add_argument('save_folder', type=str, help='Path to folder to save result in')
    parser.add_argument('--n', type=int, help='Number of images to generate', default=8)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_arguments()

    n_samples = args.n
    path = args.save_folder

    input_shape = (IMAGE_RESIZE[0],IMAGE_RESIZE[1],3)

    model_handler = ModelHandler(input_shape, N_HEADS, TIME_EMBEDDING_SIZE,
                                BETA_START, BETA_END, NOISE_STEPS,
                                load_model=True)
    model_handler.build_model()

    print("Generating samples...")

    samples = model_handler.model.sample(n_samples)

    samples = samples.numpy()

    for i, sample in enumerate(samples):

        save_path = os.path.join(path, "sample {}.jpg".format(i+1))

        image_result = Image.fromarray(sample, mode="RGB")
        image_result.save(save_path)

    print("Samples saved")
