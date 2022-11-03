# Simpsons-Diffusion-Generation

The repository is an implementation, based on **tensorflow**, of a [**diffusion**](https://arxiv.org/abs/2006.11239) model, which has been trained to generate images of faces resembling characters from the famous series **"The Simpsons"**. Sampling (and so the training) is completely unconditional, letting the model produce whatever it comes to.

## Results

Following results have been sampled after training the model for *100* epochs (more training details in the *config.py* file). Weights can be found under the *weights* folder, and they are automatically loaded when synthesizing new images. The sampling procedure is the standard one: starting from a random gaussian noise, the image is gradually denoised, thanks to the reverse diffusion process, finally being transformed into a distribution similar to the ones encountered during the training phase (in this case a *Simpsons-like* face).

<p align="center">
  <img alt="Sampling results" src="https://github.com/gaetano-signorelli/Simpsons-Diffusion-Generation/blob/main/results/Samples.png", width=640, height=320>
</p>

Despite the low resolution of *64x64* pixels (due to hardware's limitations), these results are noticeably good; especially if compared to similar generations coming from various GAN-based models. Many images reproduce closely the most relevant characters, such as the whole Simpsons family. Others create new faces, with some of them clearly reflecting a (good) combination of facial treats (or typical expressions) of different characters.

## Generate new images

The model is capable of sampling infinitely many unique (and sometimes curious) faces, with a fixed resolution of *64x64* pixels. In order to synthesize new faces, run the `generator.py` script:

`python generator.py "save_folder"`

There is only one optional argument:
- `--n` : set the number of images to sample (to be chosen accordingly to GPU's memory, default=8)

Results are saved into the specified folder as *"sample i.jpg"*, where *i* ranges from 1 to the requested number of images.

## Train the model

Training the model from scratch is as simple as editing the hyperparameters inside the *config* file. The preset that can be currently found is the one that has been adopted for the purpose of this work. Though, the model can be re-trained on whichever dataset, that should be placed inside *data/training dataset*.

The network has been trained using the dataset named *"Simpsons Faces"*, available on [Kaggle](https://www.kaggle.com/datasets/kostastokis/simpsons-faces). This one can be used to experiment further, trying to reproduce these results or to improve on them (e.g., by increasing the resolution, increasing the number of attention heads, introducing a preprocessing pipeline etc.).

To start a new training session (or recover from the last training step), run the command:

`python train.py`

Training parameters can be adjusted by accessing the file *src/model/config.py*.
