import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm

from src.model.unet import UNet

class Diffusion(Model):

    def __init__(self, input_shape, n_heads, time_embedding_size,
                beta_start, beta_end, noise_steps):

        super(Diffusion).__init__()

        self.input_shape = input_shape
        self.noise_steps = noise_steps

        self.unet_model = UNet(input_shape, n_heads, time_embedding_size)

        self.beta, self.alpha, self.alpha_hat = self.get_linear_schedules(beta_start,
                                                                        beta_end,
                                                                        noise_steps)

    def get_linear_schedules(self, start, end, steps):

        beta = tf.linspace(start, end, steps)
        alpha = 1.0 - beta
        alpha_hat = tf.math.cumprod(alpha)

        return beta, alpha, alpha_hat

    @tf.function
    def gather_and_expand(self, tensor, indices):

        gathered_tensor = tf.gather(tensor, indices)
        gathered_tensor = tf.expand_dims(gathered_tensor, axis=-1)
        gathered_tensor = tf.expand_dims(gathered_tensor, axis=-1)
        gathered_tensor = tf.expand_dims(gathered_tensor, axis=-1)

        return gathered_tensor

    @tf.function
    def noise_images(self, x, timesteps):

        gathered_alpha_hat = self.gather_and_expand(self.alpha_hat, timesteps)

        sqrt_alpha_hat = tf.math.sqrt(gathered_alpha_hat)
        sqrt_comp_alpha_hat = tf.math.sqrt(1.0-gathered_alpha_hat)

        eps = tf.random.normal(x.shape)

        gaussian_noise = sqrt_alpha_hat * x + sqrt_comp_alpha_hat * eps

        return gaussian_noise, eps

    def sample(self, n_images):

        x = tf.random.normal([n_images, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        for i in tqdm(range(self.noise_steps, 0, -1)):

            t = tf.ones(n_images) * i
            predicted_noise = self.unet_model([x, t])

            alpha = self.gather_and_expand(self.alpha, t)
            alpha_hat = self.gather_and_expand(self.alpha_hat, t)
            beta = self.gather_and_expand(self.beta, t)

            if i>1:
                noise = tf.random.normal(x.shape)

            else:
                noise = tf.zeros(x.shape)

            x = 1.0 / tf.math.sqrt(alpha) * (x - ((1.0 - alpha) / (tf.math.sqrt(1 - alpha_hat))) * predicted_noise) + tf.math.sqrt(beta) * noise

        x = (tf.clip_by_value(x, -1, 1) + 1) / 2
        x = x * 255
        x = tf.cast(x, tf.uint8)
        
        return x

    @tf.function
    def call(self, x):
        pass
